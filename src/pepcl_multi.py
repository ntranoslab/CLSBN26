# pepcl for multiple datasets
import collections
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .model_utils import PeptideMHCDataset


class PepCL_Multi:
    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps: int = 100,
        seed: int = 42,
        num_workers: int = 8,
        collate_fn=None,

        num_ms_pos: int = 50,
        ms_neg_ratio: float = 1.0,
        num_nonms_pos: int = 50,
        nonms_neg_ratio: float = 1.0,
        ms_batch_size: int = 3072,
        nonms_batch_size: int = 3072,

        auto_calibrate: bool = True,
        calibration_steps: int = 1,  # for backward compat

        continuous_target_col: str = "ba_value",
        continuous_ft_sampling_mode: str = "allele_aware",
        pooled_ft_num_pos: int = None,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        update_domain_embeddings: bool = False,

        use_replay: bool = True,
        primary_task_id: int = 1,
        sequential_backprop: bool = True,

        # -----------------------------------------------------
        # NEW: multi-domain replay configuration
        # -----------------------------------------------------
        replay_domain_ids=None,
        update_domain_id: int = 1,
        ms_replay_fraction="equal",

        ms_loss_type: str = "logsmoothap",
        ms_loss_alpha: float = 10.0,
        ms_lambda: float = 1.0,
        nonms_loss_type: str = "logsmoothap",
        nonms_loss_alpha: float = 10.0,
        ft_lambda: float = 1.0,
        dr_lambda: float = 0.0,
        dr_loss_type: str = "mse"
    ):
        self.model = model.to(device)
        self.df = train_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = device
        self.steps = steps
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.ms_loss_type = str(ms_loss_type).lower()
        self.nonms_loss_type = str(nonms_loss_type).lower()

        self.continuous_ft_loss_types = {"spearman", "mse"}
        self.ft_is_continuous = (
            self.nonms_loss_type in self.continuous_ft_loss_types
        )

        self.continuous_target_col = continuous_target_col
        self.continuous_ft_sampling_mode = continuous_ft_sampling_mode
        self.pooled_ft_num_pos = pooled_ft_num_pos

        if self.continuous_ft_sampling_mode not in {
            "allele_aware",
            "pooled"
        }:
            raise ValueError(
                "continuous_ft_sampling_mode must be one of: "
                "{'allele_aware', 'pooled'}"
            )

        if (
            self.continuous_ft_sampling_mode == "pooled"
            and not self.ft_is_continuous
        ):
            raise ValueError(
                "continuous_ft_sampling_mode='pooled' is only supported "
                "when nonms_loss_type is continuous, i.e. "
                "'spearman' or 'mse'."
            )

        if (
            self.ft_is_continuous
            and self.continuous_target_col not in self.df.columns
        ):
            raise ValueError(
                f"Continuous FT requested because "
                f"nonms_loss_type='{self.nonms_loss_type}', "
                f"but continuous_target_col="
                f"'{self.continuous_target_col}' is not in train_df."
            )

        if "domain_id" not in self.df.columns:
            raise ValueError(
                "train_df must contain a 'domain_id' column."
            )

        # =====================================================
        # Multi-domain replay configuration
        # =====================================================

        if replay_domain_ids is None:
            replay_domain_ids = [0]

        replay_domain_ids = [
            int(x) for x in replay_domain_ids
        ]

        if len(set(replay_domain_ids)) != len(replay_domain_ids):
            raise ValueError(
                "replay_domain_ids must contain unique domain IDs."
            )

        if 0 not in replay_domain_ids:
            raise ValueError(
                "Domain 0 must be included in replay_domain_ids "
                "because domain 0 is defined as MS."
            )

        self.replay_domain_ids = (
            [0] +
            [x for x in replay_domain_ids if x != 0]
        )

        self.update_domain_id = int(update_domain_id)

        if self.update_domain_id in self.replay_domain_ids:
            raise ValueError(
                "update_domain_id cannot also be in replay_domain_ids."
            )

        self.ms_replay_fraction = ms_replay_fraction

        available_domain_ids = set(
            self.df["domain_id"].astype(int).unique().tolist()
        )

        required_domain_ids = (
            set(self.replay_domain_ids)
            | {self.update_domain_id}
        )

        missing_domain_ids = (
            required_domain_ids - available_domain_ids
        )

        if missing_domain_ids:
            raise ValueError(
                f"train_df is missing required domain IDs: "
                f"{sorted(missing_domain_ids)}"
            )

        self.ms_bs = int(ms_batch_size)
        self.ft_bs = int(nonms_batch_size)

        self.replay_batch_sizes = (
            self._compute_replay_batch_sizes()
        )

        # One config per replay stream.
        # Sampling mechanism remains identical to original PepCL.
        self.replay_cfgs = {}

        for domain_id in self.replay_domain_ids:
            self.replay_cfgs[domain_id] = {
                "n_pos": num_ms_pos,
                "ratio": ms_neg_ratio,
                "bs": self.replay_batch_sizes[domain_id],
                "q_pos_by_allele": None,
                "sampling_summary": None,
                "sampling_mode": "allele_aware"
            }

        # Keep original MS config alias for convenience/backward use.
        self.ms_cfg = self.replay_cfgs[0]

        self.ft_cfg = {
            "n_pos": num_nonms_pos,
            "ratio": nonms_neg_ratio,
            "bs": self.ft_bs,
            "q_pos_by_allele": None,
            "sampling_summary": None,
            "sampling_mode": (
                self.continuous_ft_sampling_mode
                if self.ft_is_continuous
                else "allele_aware"
            ),
            "pooled_ft_num_pos": self.pooled_ft_num_pos
        }

        self.auto_calibrate = auto_calibrate
        self.calibration_steps = calibration_steps

        if self.ft_is_continuous:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True,
                feature_names=[self.continuous_target_col]
            )
        else:
            self.base_dataset = PeptideMHCDataset(
                self.df,
                self.tokenizer,
                include_domain_id=True
            )

        self.use_replay = use_replay

        # Retained exactly as the original PepCL stream selector:
        # 0 = replay stream
        # 1 = update stream
        self.primary_task_id = primary_task_id

        self.seq_backprop = sequential_backprop
        self.update_domain_embeddings = update_domain_embeddings
        self.dr_lambda = dr_lambda
        self.dr_loss_type = dr_loss_type
        self.ms_lambda = ms_lambda
        self.ft_lambda = ft_lambda

        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.criterion_ms = self._get_criterion(
            self.ms_loss_type,
            ms_loss_alpha
        )

        self.criterion_ft = self._get_criterion(
            self.nonms_loss_type,
            nonms_loss_alpha
        )

        self.teacher = None

        if self.use_replay and self.dr_lambda > 0:
            self.teacher = (
                copy.deepcopy(model)
                .to(device)
                .eval()
            )

            for p in self.teacher.parameters():
                p.requires_grad = False

        torch.manual_seed(seed)
        np.random.seed(seed)

    # =========================================================
    # Replay allocation
    # =========================================================

    @staticmethod
    def _allocate_integer_batch(weights, total):
        """
        Convert fractional domain weights into integer sample counts
        that sum exactly to total.
        """
        weights = np.asarray(weights, dtype=float)
        total = int(total)

        if total <= 0:
            return np.zeros(len(weights), dtype=int)

        if np.any(weights < 0):
            raise ValueError(
                "Replay weights cannot be negative."
            )

        if weights.sum() <= 0:
            raise ValueError(
                "Replay weights must sum to > 0."
            )

        weights = weights / weights.sum()

        raw = weights * total
        counts = np.floor(raw).astype(int)

        remaining = int(total - counts.sum())

        if remaining > 0:
            fractional = raw - counts

            # Stable so ties follow replay_domain_ids order.
            order = np.argsort(
                -fractional,
                kind="stable"
            )

            for idx in order[:remaining]:
                counts[idx] += 1

        return counts

    def _compute_replay_batch_sizes(self):
        """
        Determine exact replay samples per domain.

        ms_replay_fraction == "equal":
            All replay domains receive equal representation.

        ms_replay_fraction == float:
            Domain 0 (MS) receives that fraction.
            Remaining replay domains split the remainder equally.

        Total always equals ms_batch_size.
        """
        n_domains = len(self.replay_domain_ids)

        if n_domains == 0:
            raise ValueError(
                "At least one replay domain is required."
            )

        # Only MS in replay -> entire replay batch is MS.
        if n_domains == 1:
            return {
                self.replay_domain_ids[0]: self.ms_bs
            }

        if isinstance(self.ms_replay_fraction, str):
            if self.ms_replay_fraction.lower() != "equal":
                raise ValueError(
                    "ms_replay_fraction must be 'equal' "
                    "or a float between 0 and 1."
                )

            weights = np.ones(
                n_domains,
                dtype=float
            ) / n_domains

        else:
            ms_fraction = float(
                self.ms_replay_fraction
            )

            if not 0.0 <= ms_fraction <= 1.0:
                raise ValueError(
                    "Numeric ms_replay_fraction must be "
                    "between 0 and 1."
                )

            remaining_fraction = (
                1.0 - ms_fraction
            )

            other_fraction = (
                remaining_fraction
                / (n_domains - 1)
            )

            weights = np.array(
                [ms_fraction] +
                [other_fraction] * (n_domains - 1),
                dtype=float
            )

        counts = self._allocate_integer_batch(
            weights=weights,
            total=self.ms_bs
        )

        return {
            domain_id: int(count)
            for domain_id, count in zip(
                self.replay_domain_ids,
                counts
            )
        }

    # =========================================================
    # Losses
    # =========================================================

    class _LogSmoothAPScore(nn.Module):
        def __init__(
            self,
            alpha=10.0,
            epsilon=1e-8
        ):
            super().__init__()
            self.alpha = alpha
            self.epsilon = epsilon

        def forward(
            self,
            scores,
            labels
        ):
            y = labels.float()

            P = y.sum().clamp(min=1.0)

            diff = (
                scores.unsqueeze(0)
                - scores.unsqueeze(1)
            )

            S = torch.sigmoid(
                self.alpha * diff
            )

            num = S.matmul(y)

            den = (
                S.sum(dim=1)
                .clamp(min=self.epsilon)
            )

            soft_prec = num / den

            softap = (
                (soft_prec * y).sum()
                / P
            )

            return torch.log(
                1.0 /
                (softap + self.epsilon)
            )

    @staticmethod
    def _soft_rank(
        x: torch.Tensor,
        tau: float = 1.0
    ):
        """
        Differentiable rank based on pairwise sigmoid.

        r_i = 1 + Σ_j σ((x_j - x_i) / τ)
        """
        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(-2)

        P = torch.sigmoid(
            (x_row - x_col) / tau
        )

        return 1 + P.sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(
            self,
            tau: float = 0.1
        ):
            super().__init__()
            self.tau = tau

        def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ):
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            r_pred = PepCL_Multi._soft_rank(
                pred,
                tau=self.tau
            )

            r_target = PepCL_Multi._soft_rank(
                target,
                tau=1e-4
            )

            r_pred = (
                r_pred
                - r_pred.mean(
                    -1,
                    keepdim=True
                )
            )

            r_target = (
                r_target
                - r_target.mean(
                    -1,
                    keepdim=True
                )
            )

            num = (
                r_pred * r_target
            ).sum(-1)

            den = (
                torch.norm(
                    r_pred,
                    dim=-1
                )
                *
                torch.norm(
                    r_target,
                    dim=-1
                )
                + 1e-8
            )

            rho = num / den

            return (
                1 - rho
            ).mean()

    # =========================================================
    # Dataset wrapper
    # =========================================================

    class _TargetOverrideDataset(Dataset):
        """
        Wrapper used for the update stream.

        Classification:
            binary label is unchanged.

        Continuous:
            label is replaced by the configured continuous target.
        """

        def __init__(
            self,
            parent_ds,
            df,
            domain_id,
            continuous_target=False,
            continuous_target_col="ba_value"
        ):
            self.parent_ds = parent_ds
            self.df = df
            self.domain_id = domain_id
            self.continuous_target = continuous_target
            self.continuous_target_col = (
                continuous_target_col
            )

            self.item_continuous_key = (
                f"peptide_{continuous_target_col}"
            )

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = self.parent_ds[idx]
            item = dict(item)

            item["domain_id"] = torch.tensor(
                self.domain_id,
                dtype=torch.long
            )

            if self.continuous_target:
                if self.item_continuous_key in item:
                    target = item[
                        self.item_continuous_key
                    ]

                    if isinstance(
                        target,
                        torch.Tensor
                    ):
                        item["label"] = (
                            target.float()
                        )
                    else:
                        item["label"] = (
                            torch.tensor(
                                target,
                                dtype=torch.float
                            )
                        )

                else:
                    target = self.df.loc[
                        idx,
                        self.continuous_target_col
                    ]

                    item["label"] = torch.tensor(
                        target,
                        dtype=torch.float
                    )

            return item

    # =========================================================
    # Existing allele-aware sampler
    # =========================================================

    class _InfiniteAlleleSampler(Sampler):
        """
        Infinite allele-aware sampler.

        Same sampling behavior as the original PepCL.
        """

        def __init__(
            self,
            df,
            domain_id,
            n_pos,
            neg_ratio,
            seed,
            q_pos_by_allele=None
        ):
            self.rng = np.random.default_rng(
                seed
            )

            self.domain_id = domain_id
            self.n_pos = n_pos
            self.neg_ratio = float(
                neg_ratio
            )

            self.q_pos_by_allele = (
                q_pos_by_allele
            )

            if self.neg_ratio < 0:
                raise ValueError(
                    "neg_ratio must be >= 0. "
                    "Use 1.0 for equal negatives, "
                    "0.0 for no negatives."
                )

            self.pools = {
                0: {},
                1: {}
            }

            subset = df[
                df["domain_id"] == domain_id
            ]

            if subset.empty:
                self.alleles = []
                return

            for (
                lbl,
                allele
            ), group in subset.groupby(
                ["label", "allele"]
            ):
                lbl_int = int(lbl)

                if lbl_int not in self.pools:
                    continue

                idx = (
                    group.index
                    .to_numpy()
                )

                self.rng.shuffle(idx)

                self.pools[
                    lbl_int
                ][allele] = idx

            positive_alleles = sorted(
                list(
                    self.pools[1].keys()
                )
            )

            if (
                self.q_pos_by_allele
                is not None
            ):
                self.q_pos_by_allele = {
                    al: int(q)
                    for al, q
                    in self.q_pos_by_allele.items()
                    if int(q) > 0
                }

                self.alleles = [
                    al
                    for al
                    in positive_alleles
                    if self.q_pos_by_allele.get(
                        al,
                        0
                    ) > 0
                ]

            else:
                self.alleles = (
                    positive_alleles
                )

        def __iter__(self):
            if not self.alleles:
                return

            while True:
                indices = []

                for al in self.alleles:
                    pos_data = (
                        self.pools[1]
                        .get(al, [])
                    )

                    if len(pos_data) == 0:
                        continue

                    if (
                        self.q_pos_by_allele
                        is not None
                    ):
                        requested_pos = int(
                            self.q_pos_by_allele
                            .get(al, 0)
                        )

                    else:
                        requested_pos = int(
                            self.n_pos
                        )

                    if requested_pos <= 0:
                        continue

                    p_take = min(
                        len(pos_data),
                        requested_pos
                    )

                    if p_take <= 0:
                        continue

                    sel_pos = (
                        self.rng.choice(
                            pos_data,
                            size=p_take,
                            replace=False
                        )
                    )

                    indices.extend(sel_pos)

                    neg_data = (
                        self.pools[0]
                        .get(al, [])
                    )

                    n_req = int(
                        len(sel_pos)
                        * self.neg_ratio
                    )

                    if (
                        n_req > 0
                        and len(neg_data) > 0
                    ):
                        n_take = min(
                            len(neg_data),
                            n_req
                        )

                        if n_take > 0:
                            sel_neg = (
                                self.rng.choice(
                                    neg_data,
                                    size=n_take,
                                    replace=False
                                )
                            )

                            indices.extend(
                                sel_neg
                            )

                if len(indices) == 0:
                    return

                self.rng.shuffle(
                    indices
                )

                for i in indices:
                    yield i

    # =========================================================
    # NEW: combine domain-specific samplers into one exact batch
    # =========================================================

    class _MultiDomainReplayBatchSampler(Sampler):
        """
        Pull an exact number of samples from each independently
        allele-aware replay-domain sampler and combine them into
        one replay batch.

        This controls only replay-domain representation.
        """

        def __init__(
            self,
            domain_samplers,
            domain_batch_sizes,
            seed
        ):
            self.domain_samplers = (
                domain_samplers
            )

            self.domain_batch_sizes = {
                int(k): int(v)
                for k, v
                in domain_batch_sizes.items()
                if int(v) > 0
            }

            self.domain_ids = list(
                self.domain_batch_sizes.keys()
            )

            self.seed = seed

        def __iter__(self):
            rng = np.random.default_rng(
                self.seed
            )

            domain_iters = {
                domain_id: iter(
                    self.domain_samplers[
                        domain_id
                    ]
                )
                for domain_id
                in self.domain_ids
            }

            while True:
                batch_indices = []

                for domain_id in self.domain_ids:
                    n_samples = (
                        self.domain_batch_sizes[
                            domain_id
                        ]
                    )

                    domain_iter = (
                        domain_iters[
                            domain_id
                        ]
                    )

                    for _ in range(n_samples):
                        try:
                            idx = next(
                                domain_iter
                            )

                        except StopIteration:
                            raise RuntimeError(
                                f"Replay domain "
                                f"{domain_id} "
                                f"did not produce enough "
                                f"eligible samples."
                            )

                        batch_indices.append(
                            idx
                        )

                rng.shuffle(
                    batch_indices
                )

                yield batch_indices

    # =========================================================
    # Existing pooled continuous FT sampler
    # =========================================================

    class _InfinitePooledFTSampler(Sampler):
        def __init__(
            self,
            df,
            domain_id,
            pooled_num_pos,
            neg_ratio,
            batch_size,
            seed
        ):
            self.rng = (
                np.random.default_rng(
                    seed
                )
            )

            self.domain_id = domain_id
            self.neg_ratio = float(
                neg_ratio
            )

            self.batch_size = int(
                batch_size
            )

            if self.neg_ratio < 0:
                raise ValueError(
                    "neg_ratio must be >= 0. "
                    "Use 1.0 for equal negatives, "
                    "0.0 for no negatives."
                )

            subset = df[
                df["domain_id"] == domain_id
            ].copy()

            if subset.empty:
                self.pos_pool = np.array(
                    [],
                    dtype=int
                )

                self.neg_pool = np.array(
                    [],
                    dtype=int
                )

                self.pooled_num_pos = 0
                return

            label_int = (
                subset["label"]
                .astype(int)
            )

            self.pos_pool = (
                subset.index[
                    label_int == 1
                ].to_numpy()
            )

            self.neg_pool = (
                subset.index[
                    label_int == 0
                ].to_numpy()
            )

            if pooled_num_pos is None:
                pooled_num_pos = int(
                    self.batch_size
                    /
                    (
                        1.0
                        + self.neg_ratio
                    )
                )

                pooled_num_pos = max(
                    1,
                    pooled_num_pos
                )

            self.pooled_num_pos = int(
                pooled_num_pos
            )

            self.rng.shuffle(
                self.pos_pool
            )

            self.rng.shuffle(
                self.neg_pool
            )

        def __iter__(self):
            if (
                len(self.pos_pool) == 0
                or self.pooled_num_pos <= 0
            ):
                return

            while True:
                indices = []

                p_take = min(
                    self.pooled_num_pos,
                    len(self.pos_pool)
                )

                if p_take > 0:
                    sel_pos = (
                        self.rng.choice(
                            self.pos_pool,
                            size=p_take,
                            replace=False
                        )
                    )

                    indices.extend(
                        sel_pos
                    )

                    n_req = int(
                        len(sel_pos)
                        * self.neg_ratio
                    )

                    if (
                        n_req > 0
                        and len(self.neg_pool) > 0
                    ):
                        replace = (
                            len(self.neg_pool)
                            < n_req
                        )

                        sel_neg = (
                            self.rng.choice(
                                self.neg_pool,
                                size=n_req,
                                replace=replace
                            )
                        )

                        indices.extend(
                            sel_neg
                        )

                if len(indices) == 0:
                    return

                self.rng.shuffle(
                    indices
                )

                for i in indices:
                    yield i

    # =========================================================
    # Criteria / optimizer
    # =========================================================

    def _get_criterion(
        self,
        t,
        alpha
    ):
        t = str(t).lower()

        if t == "logsmoothap":
            return self._LogSmoothAPScore(
                alpha=alpha
            )

        if t == "spearman":
            return self._SoftSpearmanLoss(
                tau=alpha
            )

        if t == "mse":
            return nn.MSELoss()

        if t in {
            "bce",
            "bcewithlogits",
            "bcewithlogitsloss"
        }:
            return (
                nn.BCEWithLogitsLoss()
            )

        return nn.BCEWithLogitsLoss()

    def _get_dr_loss(
        self,
        student_logits,
        teacher_logits
    ):
        if self.dr_loss_type == "mse":
            return F.mse_loss(
                student_logits,
                teacher_logits
            )

        return F.mse_loss(
            student_logits,
            teacher_logits
        )

    def _get_optimizer(self):
        e = []
        d = []

        for n, p in self.model.named_parameters():
            if (
                "domain_embedder" in n
                and not self.update_domain_embeddings
            ):
                p.requires_grad = False
                continue

            if not p.requires_grad:
                continue

            if any(
                x in n
                for x in [
                    "embedder",
                    "conformer",
                    "transformer"
                ]
            ):
                e.append(p)

            else:
                d.append(p)

        return torch.optim.AdamW(
            [
                {
                    "params": e,
                    "lr": self.encoder_lr
                },
                {
                    "params": d,
                    "lr": self.decoder_lr
                }
            ],
            weight_decay=self.weight_decay
        )

    # =========================================================
    # Allele-aware quota allocation
    # =========================================================

    @staticmethod
    def _integer_quota_allocation(
        weights,
        caps,
        total,
        ensure_all_if_possible=True
    ):
        weights = np.asarray(
            weights,
            dtype=float
        )

        caps = np.asarray(
            caps,
            dtype=int
        )

        total = int(total)

        q = np.zeros_like(
            caps,
            dtype=int
        )

        valid = caps > 0

        if (
            total <= 0
            or not np.any(valid)
        ):
            return q

        max_total = int(
            min(
                total,
                caps.sum()
            )
        )

        if max_total <= 0:
            return q

        w = np.zeros_like(
            weights,
            dtype=float
        )

        if weights[valid].sum() > 0:
            w[valid] = (
                weights[valid]
                /
                weights[valid].sum()
            )

        else:
            w[valid] = (
                1.0
                / valid.sum()
            )

        remaining_total = (
            max_total
        )

        if (
            ensure_all_if_possible
            and remaining_total
            >= valid.sum()
        ):
            q[valid] = 1

            remaining_total -= int(
                valid.sum()
            )

        if remaining_total <= 0:
            return q

        while remaining_total > 0:
            capacity = caps - q

            can_receive = (
                capacity > 0
            )

            if not np.any(
                can_receive
            ):
                break

            w_active = np.zeros_like(
                w,
                dtype=float
            )

            if (
                w[can_receive].sum()
                > 0
            ):
                w_active[
                    can_receive
                ] = (
                    w[can_receive]
                    /
                    w[can_receive].sum()
                )

            else:
                w_active[
                    can_receive
                ] = (
                    1.0
                    /
                    can_receive.sum()
                )

            raw = (
                remaining_total
                * w_active
            )

            add = np.floor(
                raw
            ).astype(int)

            add = np.minimum(
                add,
                capacity
            )

            if add.sum() > 0:
                q += add

                remaining_total -= int(
                    add.sum()
                )

                if remaining_total <= 0:
                    break

            capacity = caps - q

            can_receive = (
                capacity > 0
            )

            if not np.any(
                can_receive
            ):
                break

            frac = (
                raw
                - np.floor(raw)
            )

            frac[
                ~can_receive
            ] = -1.0

            order = np.argsort(
                -frac
            )

            added_any = False

            for idx in order:
                if remaining_total <= 0:
                    break

                if capacity[idx] <= 0:
                    continue

                q[idx] += 1
                remaining_total -= 1
                added_any = True

            if not added_any:
                break

        return q

    def _compute_auto_domain_sampling(
        self,
        domain_id,
        batch_size,
        neg_ratio
    ):
        """
        Same allele-aware calibration as original PepCL,
        now callable independently for every replay domain.
        """
        if neg_ratio < 0:
            raise ValueError(
                "neg_ratio must be >= 0. "
                "Use 1.0 for equal negatives, "
                "0.0 for no negatives."
            )

        sub = self.df[
            self.df["domain_id"]
            == domain_id
        ].copy()

        if sub.empty:
            return {}, pd.DataFrame()

        sub[
            "_label_int_for_sampling"
        ] = sub["label"].astype(int)

        counts = (
            sub.groupby(
                [
                    "allele",
                    "_label_int_for_sampling"
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        if 1 in counts.columns:
            n_pos = counts[1]

        else:
            n_pos = pd.Series(
                0,
                index=counts.index
            )

        if 0 in counts.columns:
            n_neg = counts[0]

        else:
            n_neg = pd.Series(
                0,
                index=counts.index
            )

        summary = pd.DataFrame(
            {
                "allele": counts.index,
                "n_pos": (
                    n_pos
                    .astype(int)
                    .to_numpy()
                ),
                "n_neg": (
                    n_neg
                    .astype(int)
                    .to_numpy()
                )
            }
        )

        summary = (
            summary[
                summary["n_pos"] > 0
            ]
            .reset_index(
                drop=True
            )
        )

        if summary.empty:
            return {}, summary

        A = len(summary)

        P = (
            summary["n_pos"]
            .astype(float)
            .to_numpy()
        )

        caps = (
            summary["n_pos"]
            .astype(int)
            .to_numpy()
        )

        target_pos = int(
            batch_size
            /
            (
                1.0
                + float(neg_ratio)
            )
        )

        target_pos = max(
            1,
            target_pos
        )

        uniform = (
            np.ones(
                A,
                dtype=float
            )
            / A
        )

        prevalence = (
            P / P.sum()
        )

        denom = np.square(
            P
        ).sum()

        if denom <= 0:
            A_eff = float(A)

        else:
            A_eff = float(
                (P.sum() ** 2)
                / denom
            )

        balance_score = float(
            A_eff / A
        )

        balance_score = max(
            0.0,
            min(
                1.0,
                balance_score
            )
        )

        weights = (
            balance_score
            * uniform
            +
            (
                1.0
                - balance_score
            )
            * prevalence
        )

        weights = (
            weights
            / weights.sum()
        )

        q_pos = (
            self._integer_quota_allocation(
                weights=weights,
                caps=caps,
                total=target_pos,
                ensure_all_if_possible=False
            )
        )

        q_pos_by_allele = {
            al: int(q)
            for al, q
            in zip(
                summary[
                    "allele"
                ].tolist(),
                q_pos
            )
            if int(q) > 0
        }

        summary[
            "allele_weight"
        ] = weights

        summary[
            "q_pos"
        ] = q_pos.astype(int)

        summary[
            "expected_q_neg_if_available"
        ] = np.floor(
            summary[
                "q_pos"
            ].to_numpy()
            *
            float(neg_ratio)
        ).astype(int)

        summary[
            "reuse_pressure_pos"
        ] = (
            summary["q_pos"]
            /
            summary[
                "n_pos"
            ].clip(lower=1)
        )

        summary[
            "has_negatives"
        ] = (
            summary["n_neg"] > 0
        )

        summary.attrs[
            "target_pos"
        ] = target_pos

        summary.attrs[
            "n_alleles"
        ] = A

        summary.attrs[
            "effective_n_alleles"
        ] = A_eff

        summary.attrs[
            "balance_score"
        ] = balance_score

        return (
            q_pos_by_allele,
            summary
        )

    # =========================================================
    # Continuous FT summary
    # =========================================================

    def _compute_pooled_ft_summary(self):
        sub = self.df[
            self.df["domain_id"]
            == self.update_domain_id
        ].copy()

        if sub.empty:
            return pd.DataFrame()

        sub[
            "_label_int_for_sampling"
        ] = sub["label"].astype(int)

        n_pos_total = int(
            (
                sub[
                    "_label_int_for_sampling"
                ] == 1
            ).sum()
        )

        n_neg_total = int(
            (
                sub[
                    "_label_int_for_sampling"
                ] == 0
            ).sum()
        )

        n_alleles_pos = int(
            sub.loc[
                sub[
                    "_label_int_for_sampling"
                ] == 1,
                "allele"
            ].nunique()
        )

        n_alleles_all = int(
            sub["allele"].nunique()
        )

        if self.pooled_ft_num_pos is None:
            pooled_pos = int(
                self.ft_cfg["bs"]
                /
                (
                    1.0
                    + float(
                        self.ft_cfg[
                            "ratio"
                        ]
                    )
                )
            )

            pooled_pos = max(
                1,
                pooled_pos
            )

        else:
            pooled_pos = int(
                self.pooled_ft_num_pos
            )

        pooled_neg = int(
            pooled_pos
            *
            float(
                self.ft_cfg["ratio"]
            )
        )

        summary = pd.DataFrame(
            {
                "mode": ["pooled"],
                "n_pos_total": [
                    n_pos_total
                ],
                "n_neg_total": [
                    n_neg_total
                ],
                "n_alleles_pos": [
                    n_alleles_pos
                ],
                "n_alleles_all": [
                    n_alleles_all
                ],
                "pooled_ft_num_pos": [
                    pooled_pos
                ],
                "expected_pooled_num_neg": [
                    pooled_neg
                ],
                "pos_reuse_pressure": [
                    pooled_pos
                    / max(
                        n_pos_total,
                        1
                    )
                ]
            }
        )

        summary.attrs[
            "pooled_ft_num_pos"
        ] = pooled_pos

        summary.attrs[
            "expected_pooled_num_neg"
        ] = pooled_neg

        return summary

    # =========================================================
    # Auto-calibration
    # =========================================================

    def _auto_calibrate(self):
        """
        Replay:
            calibrate allele-aware sampling independently
            inside every replay domain.

        Update:
            identical behavior to original PepCL.
        """

        for (
            domain_id,
            cfg
        ) in self.replay_cfgs.items():

            # A zero-weight domain is simply skipped.
            if cfg["bs"] <= 0:
                cfg[
                    "q_pos_by_allele"
                ] = None

                cfg[
                    "sampling_summary"
                ] = pd.DataFrame()

                continue

            q_pos, summary = (
                self._compute_auto_domain_sampling(
                    domain_id=domain_id,
                    batch_size=cfg["bs"],
                    neg_ratio=cfg["ratio"]
                )
            )

            cfg[
                "q_pos_by_allele"
            ] = q_pos

            cfg[
                "sampling_summary"
            ] = summary

        if (
            self.ft_is_continuous
            and
            self.continuous_ft_sampling_mode
            == "pooled"
        ):
            ft_summary = (
                self._compute_pooled_ft_summary()
            )

            self.ft_cfg[
                "q_pos_by_allele"
            ] = None

            self.ft_cfg[
                "sampling_summary"
            ] = ft_summary

        else:
            (
                ft_q,
                ft_summary
            ) = (
                self._compute_auto_domain_sampling(
                    domain_id=(
                        self.update_domain_id
                    ),
                    batch_size=(
                        self.ft_cfg["bs"]
                    ),
                    neg_ratio=(
                        self.ft_cfg["ratio"]
                    )
                )
            )

            self.ft_cfg[
                "q_pos_by_allele"
            ] = ft_q

            self.ft_cfg[
                "sampling_summary"
            ] = ft_summary

        def _summary_line(
            name,
            summary,
            cfg
        ):
            if (
                summary is None
                or summary.empty
            ):
                return (
                    f"  {name}: "
                    f"no eligible positive alleles"
                )

            if (
                "mode" in summary.columns
                and
                summary[
                    "mode"
                ].iloc[0]
                == "pooled"
            ):
                row = summary.iloc[0]

                return (
                    f"  {name}: pooled continuous FT, "
                    f"alleles_pos="
                    f"{int(row['n_alleles_pos'])}, "
                    f"alleles_all="
                    f"{int(row['n_alleles_all'])}, "
                    f"pos_pool="
                    f"{int(row['n_pos_total'])}, "
                    f"neg_pool="
                    f"{int(row['n_neg_total'])}, "
                    f"pooled_pos/cycle="
                    f"{int(row['pooled_ft_num_pos'])}, "
                    f"expected_neg/cycle="
                    f"{int(row['expected_pooled_num_neg'])}, "
                    f"pos_reuse="
                    f"{float(row['pos_reuse_pressure']):.4f}, "
                    f"neg_ratio={cfg['ratio']}"
                )

            target_pos = summary.attrs.get(
                "target_pos",
                None
            )

            A = summary.attrs.get(
                "n_alleles",
                len(summary)
            )

            A_eff = summary.attrs.get(
                "effective_n_alleles",
                np.nan
            )

            balance_score = summary.attrs.get(
                "balance_score",
                np.nan
            )

            q_nonzero = summary.loc[
                summary["q_pos"] > 0,
                "q_pos"
            ]

            reuse = summary.loc[
                summary["q_pos"] > 0,
                "reuse_pressure_pos"
            ]

            q_med = (
                float(
                    q_nonzero.median()
                )
                if len(q_nonzero)
                else 0.0
            )

            q_min = (
                int(
                    q_nonzero.min()
                )
                if len(q_nonzero)
                else 0
            )

            q_max = (
                int(
                    q_nonzero.max()
                )
                if len(q_nonzero)
                else 0
            )

            reuse_med = (
                float(
                    reuse.median()
                )
                if len(reuse)
                else 0.0
            )

            reuse_max = (
                float(
                    reuse.max()
                )
                if len(reuse)
                else 0.0
            )

            return (
                f"  {name}: "
                f"batch={cfg['bs']}, "
                f"alleles={A}, "
                f"A_eff={A_eff:.2f}, "
                f"balance={balance_score:.3f}, "
                f"target_pos/cycle={target_pos}, "
                f"q_pos median={q_med:.1f} "
                f"[{q_min}, {q_max}], "
                f"reuse median={reuse_med:.4f}, "
                f"reuse max={reuse_max:.4f}, "
                f"neg_ratio={cfg['ratio']}"
            )

        print(
            "[PepCL_Multi] Replay batch allocation:"
        )

        for domain_id in self.replay_domain_ids:
            label = (
                "MS"
                if domain_id == 0
                else f"domain {domain_id}"
            )

            print(
                f"  {label}: "
                f"{self.replay_batch_sizes[domain_id]} "
                f"/ {self.ms_bs}"
            )

        print(
            "\n[PepCL_Multi] "
            "Auto-calibrated sampling:"
        )

        for domain_id in self.replay_domain_ids:
            cfg = self.replay_cfgs[
                domain_id
            ]

            label = (
                "Replay MS (domain 0)"
                if domain_id == 0
                else
                f"Replay domain {domain_id}"
            )

            print(
                _summary_line(
                    label,
                    cfg[
                        "sampling_summary"
                    ],
                    cfg
                )
            )

        print(
            _summary_line(
                f"FT domain "
                f"{self.update_domain_id}",
                self.ft_cfg[
                    "sampling_summary"
                ],
                self.ft_cfg
            )
        )

        print("")

    # =========================================================
    # Training
    # =========================================================

    def train(self):
        if self.auto_calibrate:
            self._auto_calibrate()

        else:
            if (
                self.ft_is_continuous
                and
                self.continuous_ft_sampling_mode
                == "pooled"
            ):
                self.ft_cfg[
                    "sampling_summary"
                ] = (
                    self._compute_pooled_ft_summary()
                )

            print(
                "[PepCL_Multi] Manual mode"
            )

            print(
                "[PepCL_Multi] Replay batch allocation:"
            )

            for domain_id in self.replay_domain_ids:
                print(
                    f"  domain {domain_id}: "
                    f"{self.replay_batch_sizes[domain_id]} "
                    f"/ {self.ms_bs}"
                )

            print(
                f"  Replay n_pos="
                f"{self.replay_cfgs[0]['n_pos']}, "
                f"Replay neg_ratio="
                f"{self.replay_cfgs[0]['ratio']} | "
                f"FT n_pos="
                f"{self.ft_cfg['n_pos']}, "
                f"FT neg_ratio="
                f"{self.ft_cfg['ratio']} | "
                f"FT continuous="
                f"{self.ft_is_continuous}, "
                f"FT sampling_mode="
                f"{self.ft_cfg['sampling_mode']}\n"
            )

        # -----------------------------------------------------
        # Build multi-domain replay loader
        # -----------------------------------------------------

        def _build_replay_loader():
            domain_samplers = {}

            active_batch_sizes = {}

            for domain_id in self.replay_domain_ids:
                cfg = self.replay_cfgs[
                    domain_id
                ]

                if cfg["bs"] <= 0:
                    continue

                sampler = (
                    self._InfiniteAlleleSampler(
                        self.df,
                        domain_id=domain_id,
                        n_pos=cfg["n_pos"],
                        neg_ratio=cfg["ratio"],
                        seed=(
                            self.seed
                            + domain_id
                        ),
                        q_pos_by_allele=cfg.get(
                            "q_pos_by_allele",
                            None
                        )
                    )
                )

                if not sampler.alleles:
                    raise ValueError(
                        f"Replay domain {domain_id} "
                        f"contains no eligible positive "
                        f"alleles."
                    )

                domain_samplers[
                    domain_id
                ] = sampler

                active_batch_sizes[
                    domain_id
                ] = cfg["bs"]

            if not domain_samplers:
                raise ValueError(
                    "No active replay domains were "
                    "available to construct a replay batch."
                )

            batch_sampler = (
                self._MultiDomainReplayBatchSampler(
                    domain_samplers=(
                        domain_samplers
                    ),
                    domain_batch_sizes=(
                        active_batch_sizes
                    ),
                    seed=self.seed
                )
            )

            return DataLoader(
                self.base_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=(
                    self.num_workers > 0
                )
            )

        # -----------------------------------------------------
        # Build update loader
        # -----------------------------------------------------

        def _build_ft_loader():
            ds = (
                self._TargetOverrideDataset(
                    parent_ds=self.base_dataset,
                    df=self.df,
                    domain_id=(
                        self.update_domain_id
                    ),
                    continuous_target=(
                        self.ft_is_continuous
                    ),
                    continuous_target_col=(
                        self.continuous_target_col
                    )
                )
            )

            if (
                self.ft_is_continuous
                and
                self.ft_cfg.get(
                    "sampling_mode",
                    "allele_aware"
                ) == "pooled"
            ):
                sampler = (
                    self._InfinitePooledFTSampler(
                        self.df,
                        domain_id=(
                            self.update_domain_id
                        ),
                        pooled_num_pos=(
                            self.ft_cfg.get(
                                "pooled_ft_num_pos",
                                None
                            )
                        ),
                        neg_ratio=(
                            self.ft_cfg["ratio"]
                        ),
                        batch_size=(
                            self.ft_cfg["bs"]
                        ),
                        seed=(
                            self.seed + 10000
                        )
                    )
                )

            else:
                sampler = (
                    self._InfiniteAlleleSampler(
                        self.df,
                        domain_id=(
                            self.update_domain_id
                        ),
                        n_pos=(
                            self.ft_cfg["n_pos"]
                        ),
                        neg_ratio=(
                            self.ft_cfg["ratio"]
                        ),
                        seed=(
                            self.seed + 10000
                        ),
                        q_pos_by_allele=(
                            self.ft_cfg.get(
                                "q_pos_by_allele",
                                None
                            )
                        )
                    )
                )

            return DataLoader(
                ds,
                batch_size=(
                    self.ft_cfg["bs"]
                ),
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=(
                    self.num_workers > 0
                )
            )

        need_replay_stream = (
            self.use_replay
            or self.primary_task_id == 0
        )

        need_ft_stream = (
            self.use_replay
            or self.primary_task_id == 1
        )

        replay_iter = None
        ft_iter = None

        if need_replay_stream:
            replay_iter = iter(
                _build_replay_loader()
            )

        if need_ft_stream:
            ft_iter = iter(
                _build_ft_loader()
            )

        optimizer = self._get_optimizer()

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                int(
                    self.steps
                    * self.warmup_ratio
                ),
                self.steps
            )
        )

        scaler = GradScaler()

        metrics = collections.defaultdict(
            list
        )

        pbar = tqdm(
            range(self.steps),
            desc="PepCL_Multi"
        )

        for step in pbar:
            self.model.train()
            optimizer.zero_grad()

            l_ms_v = 0.0
            l_ft_v = 0.0
            l_dr_v = 0.0

            loss_ms = None
            loss_ft = None

            # =================================================
            # Replay stream
            #
            # This is the same PepCL replay loss as before.
            # The only difference is that the batch can contain
            # multiple replay domains at controlled proportions.
            # =================================================

            if need_replay_stream:
                try:
                    batch_ms = next(
                        replay_iter
                    )

                    batch_ms = {
                        k: (
                            v.to(self.device)
                            if isinstance(
                                v,
                                torch.Tensor
                            )
                            else v
                        )
                        for k, v
                        in batch_ms.items()
                    }

                    lbl_ms = batch_ms.pop(
                        "label"
                    )

                    with autocast(
                        self.device
                    ):
                        logits_ms = (
                            self.model(
                                batch_ms
                            )
                            .view(-1)
                        )

                        loss_ms = (
                            self.criterion_ms(
                                logits_ms,
                                lbl_ms
                            )
                            *
                            self.ms_lambda
                        )

                        if (
                            self.dr_lambda > 0
                            and
                            self.teacher is not None
                        ):
                            with torch.no_grad():
                                t_logits = (
                                    self.teacher(
                                        batch_ms
                                    )
                                    .view(-1)
                                )

                            loss_dr = (
                                self._get_dr_loss(
                                    logits_ms,
                                    t_logits
                                )
                                *
                                self.dr_lambda
                            )

                            loss_ms += loss_dr

                            l_dr_v = (
                                loss_dr.item()
                            )

                    if self.seq_backprop:
                        scaler.scale(
                            loss_ms
                        ).backward()

                    l_ms_v = (
                        loss_ms.item()
                    )

                except (
                    StopIteration,
                    AttributeError
                ):
                    pass

            # =================================================
            # Update stream
            # Identical to original PepCL
            # =================================================

            if need_ft_stream:
                try:
                    batch_ft = next(
                        ft_iter
                    )

                    batch_ft = {
                        k: (
                            v.to(self.device)
                            if isinstance(
                                v,
                                torch.Tensor
                            )
                            else v
                        )
                        for k, v
                        in batch_ft.items()
                    }

                    lbl_ft = batch_ft.pop(
                        "label"
                    )

                    with autocast(
                        self.device
                    ):
                        logits_ft = (
                            self.model(
                                batch_ft
                            )
                            .view(-1)
                        )

                        loss_ft = (
                            self.criterion_ft(
                                logits_ft,
                                lbl_ft
                            )
                            *
                            self.ft_lambda
                        )

                    if self.seq_backprop:
                        scaler.scale(
                            loss_ft
                        ).backward()

                    l_ft_v = (
                        loss_ft.item()
                    )

                except (
                    StopIteration,
                    AttributeError
                ):
                    pass

            # =================================================
            # Non-sequential backprop
            # Identical to original PepCL
            # =================================================

            if not self.seq_backprop:
                total_l = None

                if loss_ms is not None:
                    total_l = loss_ms

                if loss_ft is not None:
                    total_l = (
                        loss_ft
                        if total_l is None
                        else total_l
                        + loss_ft
                    )

                if total_l is not None:
                    scaler.scale(
                        total_l
                    ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            pbar.set_postfix(
                {
                    "Replay": (
                        f"{l_ms_v:.6f}"
                    ),
                    "FT": (
                        f"{l_ft_v:.6f}"
                    ),
                    "DR": (
                        f"{l_dr_v:.6f}"
                    )
                }
            )

            # Keep original metric names for compatibility.
            metrics[
                "ms_loss"
            ].append(
                l_ms_v
            )

            metrics[
                "ft_loss"
            ].append(
                l_ft_v
            )

            metrics[
                "dr_loss"
            ].append(
                l_dr_v
            )

            if (
                self.output_dir
                and
                (step + 1)
                % self.save_every_n_steps
                == 0
            ):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.output_dir,
                        f"pepcl_step_"
                        f"{step + 1}.pt"
                    )
                )

        return metrics


# version 2 allows list of domains for the update domain ids as well to do a single update

class PepCL_Multi_V2:
    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps: int = 100,
        seed: int = 42,
        num_workers: int = 8,
        collate_fn=None,

        num_ms_pos: int = 50,
        ms_neg_ratio: float = 1.0,
        num_nonms_pos: int = 50,
        nonms_neg_ratio: float = 1.0,
        ms_batch_size: int = 3072,
        nonms_batch_size: int = 3072,

        auto_calibrate: bool = True,
        calibration_steps: int = 1,

        continuous_target_col: str = "ba_value",
        continuous_ft_sampling_mode: str = "allele_aware",
        pooled_ft_num_pos: int = None,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        update_domain_embeddings: bool = False,

        use_replay: bool = True,
        primary_task_id: int = 1,
        sequential_backprop: bool = True,

        # Multi-domain replay
        replay_domain_ids=None,
        ms_replay_fraction="equal",

        # Multi-domain update
        update_domain_id: int = 1,
        update_domain_ids=None,

        ms_loss_type: str = "logsmoothap",
        ms_loss_alpha: float = 10.0,
        ms_lambda: float = 1.0,
        nonms_loss_type: str = "logsmoothap",
        nonms_loss_alpha: float = 10.0,
        ft_lambda: float = 1.0,
        dr_lambda: float = 0.0,
        dr_loss_type: str = "mse"
    ):
        self.model = model.to(device)
        self.df = train_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.device = device
        self.steps = steps
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.ms_loss_type = str(ms_loss_type).lower()
        self.nonms_loss_type = str(nonms_loss_type).lower()
        self.continuous_ft_loss_types = {"spearman", "mse"}
        self.ft_is_continuous = self.nonms_loss_type in self.continuous_ft_loss_types
        self.continuous_target_col = continuous_target_col
        self.continuous_ft_sampling_mode = continuous_ft_sampling_mode
        self.pooled_ft_num_pos = pooled_ft_num_pos

        if self.continuous_ft_sampling_mode not in {"allele_aware", "pooled"}:
            raise ValueError("continuous_ft_sampling_mode must be one of {'allele_aware', 'pooled'}.")

        if self.continuous_ft_sampling_mode == "pooled" and not self.ft_is_continuous:
            raise ValueError("continuous_ft_sampling_mode='pooled' requires nonms_loss_type='spearman' or 'mse'.")

        if self.ft_is_continuous and self.continuous_target_col not in self.df.columns:
            raise ValueError(
                f"Continuous FT requested because nonms_loss_type='{self.nonms_loss_type}', "
                f"but '{self.continuous_target_col}' is not in train_df."
            )

        if "domain_id" not in self.df.columns:
            raise ValueError("train_df must contain a 'domain_id' column.")

        # -----------------------------------------------------
        # Domain configuration
        # -----------------------------------------------------
        if replay_domain_ids is None:
            replay_domain_ids = [0]

        self.replay_domain_ids = [int(x) for x in replay_domain_ids]
        if len(set(self.replay_domain_ids)) != len(self.replay_domain_ids):
            raise ValueError("replay_domain_ids must contain unique domain IDs.")
        if 0 not in self.replay_domain_ids:
            raise ValueError("Domain 0 must be included in replay_domain_ids because domain 0 is MS.")

        # Keep MS first for deterministic allocation.
        self.replay_domain_ids = [0] + [x for x in self.replay_domain_ids if x != 0]

        # Backward compatible: scalar update_domain_id still works.
        if update_domain_ids is None:
            self.update_domain_ids = [int(update_domain_id)]
        else:
            self.update_domain_ids = [int(x) for x in update_domain_ids]

        if not self.update_domain_ids:
            raise ValueError("At least one update domain is required.")
        if len(set(self.update_domain_ids)) != len(self.update_domain_ids):
            raise ValueError("update_domain_ids must contain unique domain IDs.")

        available_domains = set(self.df["domain_id"].astype(int).unique())
        required_domains = set(self.replay_domain_ids) | set(self.update_domain_ids)
        missing_domains = required_domains - available_domains
        if missing_domains:
            raise ValueError(f"train_df is missing required domain IDs: {sorted(missing_domains)}")

        self.ms_replay_fraction = ms_replay_fraction
        self.ms_bs = int(ms_batch_size)
        self.ft_bs = int(nonms_batch_size)

        # Exact batch contribution from each replay/update domain.
        self.replay_batch_sizes = self._compute_replay_batch_sizes()
        self.update_batch_sizes = self._compute_equal_batch_sizes(self.update_domain_ids, self.ft_bs)

        # One independently calibrated config per replay domain.
        self.replay_cfgs = {
            d: {
                "n_pos": num_ms_pos,
                "ratio": ms_neg_ratio,
                "bs": self.replay_batch_sizes[d],
                "q_pos_by_allele": None,
                "sampling_summary": None,
                "sampling_mode": "allele_aware"
            }
            for d in self.replay_domain_ids
        }

        # One independently calibrated config per update domain.
        self.ft_cfgs = {
            d: {
                "n_pos": num_nonms_pos,
                "ratio": nonms_neg_ratio,
                "bs": self.update_batch_sizes[d],
                "q_pos_by_allele": None,
                "sampling_summary": None,
                "sampling_mode": (
                    self.continuous_ft_sampling_mode
                    if self.ft_is_continuous else "allele_aware"
                ),
                "pooled_ft_num_pos": self.pooled_ft_num_pos
            }
            for d in self.update_domain_ids
        }

        # Aliases retained for compatibility with prior code.
        self.ms_cfg = self.replay_cfgs[0]
        self.ft_cfg = self.ft_cfgs[self.update_domain_ids[0]]

        self.auto_calibrate = auto_calibrate
        self.calibration_steps = calibration_steps

        if self.ft_is_continuous:
            self.base_dataset = PeptideMHCDataset(
                self.df, self.tokenizer, include_domain_id=True,
                feature_names=[self.continuous_target_col]
            )
        else:
            self.base_dataset = PeptideMHCDataset(
                self.df, self.tokenizer, include_domain_id=True
            )

        self.use_replay = use_replay
        self.primary_task_id = primary_task_id
        self.seq_backprop = sequential_backprop
        self.update_domain_embeddings = update_domain_embeddings
        self.dr_lambda = dr_lambda
        self.dr_loss_type = dr_loss_type
        self.ms_lambda = ms_lambda
        self.ft_lambda = ft_lambda

        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        self.teacher = None
        if self.use_replay and self.dr_lambda > 0:
            self.teacher = copy.deepcopy(model).to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        torch.manual_seed(seed)
        np.random.seed(seed)

    # =========================================================
    # Batch allocation
    # =========================================================

    @staticmethod
    def _allocate_integer_batch(weights, total):
        weights = np.asarray(weights, dtype=float)
        total = int(total)

        if total <= 0:
            return np.zeros(len(weights), dtype=int)
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Batch-allocation weights must be nonnegative and sum to > 0.")

        weights /= weights.sum()
        raw = weights * total
        counts = np.floor(raw).astype(int)
        remaining = total - counts.sum()

        if remaining > 0:
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[:remaining]] += 1

        return counts

    @classmethod
    def _compute_equal_batch_sizes(cls, domain_ids, total):
        counts = cls._allocate_integer_batch(np.ones(len(domain_ids)), total)
        return {d: int(n) for d, n in zip(domain_ids, counts)}

    def _compute_replay_batch_sizes(self):
        n = len(self.replay_domain_ids)

        if n == 1:
            return {self.replay_domain_ids[0]: self.ms_bs}

        if isinstance(self.ms_replay_fraction, str):
            if self.ms_replay_fraction.lower() != "equal":
                raise ValueError("ms_replay_fraction must be 'equal' or a float between 0 and 1.")
            weights = np.ones(n)
        else:
            ms_fraction = float(self.ms_replay_fraction)
            if not 0 <= ms_fraction <= 1:
                raise ValueError("Numeric ms_replay_fraction must be between 0 and 1.")

            other_fraction = (1.0 - ms_fraction) / (n - 1)
            weights = np.array([ms_fraction] + [other_fraction] * (n - 1))

        counts = self._allocate_integer_batch(weights, self.ms_bs)
        return {d: int(n) for d, n in zip(self.replay_domain_ids, counts)}

    # =========================================================
    # Losses
    # =========================================================

    class _LogSmoothAPScore(nn.Module):
        def __init__(self, alpha=10.0, epsilon=1e-8):
            super().__init__()
            self.alpha = alpha
            self.epsilon = epsilon

        def forward(self, scores, labels):
            y = labels.float()
            P = y.sum().clamp(min=1.0)
            diff = scores.unsqueeze(0) - scores.unsqueeze(1)
            S = torch.sigmoid(self.alpha * diff)
            num = S.matmul(y)
            den = S.sum(dim=1).clamp(min=self.epsilon)
            soft_prec = num / den
            softap = (soft_prec * y).sum() / P
            return torch.log(1.0 / (softap + self.epsilon))

    @staticmethod
    def _soft_rank(x: torch.Tensor, tau: float = 1.0):
        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(-2)
        return 1 + torch.sigmoid((x_row - x_col) / tau).sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(self, tau: float = 0.1):
            super().__init__()
            self.tau = tau

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            r_pred = PepCL_Multi._soft_rank(pred, tau=self.tau)
            r_target = PepCL_Multi._soft_rank(target, tau=1e-4)

            r_pred -= r_pred.mean(-1, keepdim=True)
            r_target -= r_target.mean(-1, keepdim=True)

            num = (r_pred * r_target).sum(-1)
            den = torch.norm(r_pred, dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
            return (1 - num / den).mean()

    # =========================================================
    # Target wrapper
    # =========================================================

    class _TargetOverrideDataset(Dataset):
        """
        Preserves each row's actual domain_id.

        For continuous FT only, replaces the binary label with
        continuous_target_col. For classification, labels remain unchanged.
        """
        def __init__(self, parent_ds, df, continuous_target=False, continuous_target_col="ba_value"):
            self.parent_ds = parent_ds
            self.df = df
            self.continuous_target = continuous_target
            self.continuous_target_col = continuous_target_col
            self.item_continuous_key = f"peptide_{continuous_target_col}"

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = dict(self.parent_ds[idx])

            if self.continuous_target:
                if self.item_continuous_key in item:
                    target = item[self.item_continuous_key]
                    item["label"] = target.float() if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float)
                else:
                    item["label"] = torch.tensor(self.df.loc[idx, self.continuous_target_col], dtype=torch.float)

            return item

    # =========================================================
    # Allele-aware sampler
    # =========================================================

    class _InfiniteAlleleSampler(Sampler):
        def __init__(self, df, domain_id, n_pos, neg_ratio, seed, q_pos_by_allele=None):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.n_pos = n_pos
            self.neg_ratio = float(neg_ratio)
            self.q_pos_by_allele = q_pos_by_allele

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0.")

            self.pools = {0: {}, 1: {}}
            subset = df[df["domain_id"] == domain_id]

            if subset.empty:
                self.alleles = []
                return

            for (lbl, allele), group in subset.groupby(["label", "allele"]):
                lbl_int = int(lbl)
                if lbl_int not in self.pools:
                    continue
                idx = group.index.to_numpy()
                self.rng.shuffle(idx)
                self.pools[lbl_int][allele] = idx

            positive_alleles = sorted(self.pools[1].keys())

            if self.q_pos_by_allele is not None:
                self.q_pos_by_allele = {
                    al: int(q) for al, q in self.q_pos_by_allele.items() if int(q) > 0
                }
                self.alleles = [
                    al for al in positive_alleles if self.q_pos_by_allele.get(al, 0) > 0
                ]
            else:
                self.alleles = positive_alleles

        def __iter__(self):
            if not self.alleles:
                return

            while True:
                indices = []

                for al in self.alleles:
                    pos_data = self.pools[1].get(al, [])
                    if len(pos_data) == 0:
                        continue

                    requested_pos = (
                        int(self.q_pos_by_allele.get(al, 0))
                        if self.q_pos_by_allele is not None
                        else int(self.n_pos)
                    )

                    if requested_pos <= 0:
                        continue

                    p_take = min(len(pos_data), requested_pos)
                    if p_take <= 0:
                        continue

                    sel_pos = self.rng.choice(pos_data, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    neg_data = self.pools[0].get(al, [])
                    n_req = int(len(sel_pos) * self.neg_ratio)

                    if n_req > 0 and len(neg_data) > 0:
                        n_take = min(len(neg_data), n_req)
                        if n_take > 0:
                            indices.extend(self.rng.choice(neg_data, size=n_take, replace=False))

                if not indices:
                    return

                self.rng.shuffle(indices)
                yield from indices

    # =========================================================
    # Multi-domain batch sampler
    # =========================================================

    class _MultiDomainBatchSampler(Sampler):
        """
        Combines independently sampled domains into one exact minibatch.
        Used identically for replay and multi-domain FT streams.
        """
        def __init__(self, domain_samplers, domain_batch_sizes, seed):
            self.domain_samplers = domain_samplers
            self.domain_batch_sizes = {
                int(k): int(v) for k, v in domain_batch_sizes.items() if int(v) > 0
            }
            self.domain_ids = list(self.domain_batch_sizes)
            self.seed = seed

        def __iter__(self):
            rng = np.random.default_rng(self.seed)
            domain_iters = {d: iter(self.domain_samplers[d]) for d in self.domain_ids}

            while True:
                batch_indices = []

                for d in self.domain_ids:
                    for _ in range(self.domain_batch_sizes[d]):
                        try:
                            batch_indices.append(next(domain_iters[d]))
                        except StopIteration:
                            raise RuntimeError(f"Domain {d} did not produce enough eligible samples.")

                rng.shuffle(batch_indices)
                yield batch_indices

    # =========================================================
    # Original pooled continuous FT sampler
    # =========================================================

    class _InfinitePooledFTSampler(Sampler):
        def __init__(self, df, domain_id, pooled_num_pos, neg_ratio, batch_size, seed):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.neg_ratio = float(neg_ratio)
            self.batch_size = int(batch_size)

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0.")

            subset = df[df["domain_id"] == domain_id].copy()

            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                self.pooled_num_pos = 0
                return

            labels = subset["label"].astype(int)
            self.pos_pool = subset.index[labels == 1].to_numpy()
            self.neg_pool = subset.index[labels == 0].to_numpy()

            if pooled_num_pos is None:
                pooled_num_pos = max(1, int(self.batch_size / (1.0 + self.neg_ratio)))

            self.pooled_num_pos = int(pooled_num_pos)
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        def __iter__(self):
            if len(self.pos_pool) == 0 or self.pooled_num_pos <= 0:
                return

            while True:
                indices = []
                p_take = min(self.pooled_num_pos, len(self.pos_pool))

                if p_take > 0:
                    sel_pos = self.rng.choice(self.pos_pool, size=p_take, replace=False)
                    indices.extend(sel_pos)

                    n_req = int(len(sel_pos) * self.neg_ratio)
                    if n_req > 0 and len(self.neg_pool) > 0:
                        replace = len(self.neg_pool) < n_req
                        indices.extend(self.rng.choice(self.neg_pool, size=n_req, replace=replace))

                if not indices:
                    return

                self.rng.shuffle(indices)
                yield from indices

    # =========================================================
    # Criterion / DR / optimizer
    # =========================================================

    def _get_criterion(self, t, alpha):
        t = str(t).lower()
        if t == "logsmoothap":
            return self._LogSmoothAPScore(alpha=alpha)
        if t == "spearman":
            return self._SoftSpearmanLoss(tau=alpha)
        if t == "mse":
            return nn.MSELoss()
        return nn.BCEWithLogitsLoss()

    def _get_dr_loss(self, student_logits, teacher_logits):
        return F.mse_loss(student_logits, teacher_logits)

    def _get_optimizer(self):
        e, d = [], []

        for n, p in self.model.named_parameters():
            if "domain_embedder" in n and not self.update_domain_embeddings:
                p.requires_grad = False
                continue
            if not p.requires_grad:
                continue

            if any(x in n for x in ["embedder", "conformer", "transformer"]):
                e.append(p)
            else:
                d.append(p)

        return torch.optim.AdamW(
            [{"params": e, "lr": self.encoder_lr}, {"params": d, "lr": self.decoder_lr}],
            weight_decay=self.weight_decay
        )

    # =========================================================
    # Allele quota calibration
    # =========================================================

    @staticmethod
    def _integer_quota_allocation(weights, caps, total, ensure_all_if_possible=True):
        weights = np.asarray(weights, dtype=float)
        caps = np.asarray(caps, dtype=int)
        total = int(total)
        q = np.zeros_like(caps, dtype=int)

        valid = caps > 0
        if total <= 0 or not np.any(valid):
            return q

        remaining_total = int(min(total, caps.sum()))
        if remaining_total <= 0:
            return q

        w = np.zeros_like(weights, dtype=float)
        w[valid] = (
            weights[valid] / weights[valid].sum()
            if weights[valid].sum() > 0
            else 1.0 / valid.sum()
        )

        if ensure_all_if_possible and remaining_total >= valid.sum():
            q[valid] = 1
            remaining_total -= int(valid.sum())

        while remaining_total > 0:
            capacity = caps - q
            active = capacity > 0

            if not np.any(active):
                break

            w_active = np.zeros_like(w)
            w_active[active] = (
                w[active] / w[active].sum()
                if w[active].sum() > 0
                else 1.0 / active.sum()
            )

            raw = remaining_total * w_active
            add = np.minimum(np.floor(raw).astype(int), capacity)

            if add.sum() > 0:
                q += add
                remaining_total -= int(add.sum())

            if remaining_total <= 0:
                break

            capacity = caps - q
            active = capacity > 0

            if not np.any(active):
                break

            frac = raw - np.floor(raw)
            frac[~active] = -1.0

            added_any = False
            for idx in np.argsort(-frac):
                if remaining_total <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                q[idx] += 1
                remaining_total -= 1
                added_any = True

            if not added_any:
                break

        return q

    def _compute_auto_domain_sampling(self, domain_id, batch_size, neg_ratio):
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be >= 0.")

        sub = self.df[self.df["domain_id"] == domain_id].copy()
        if sub.empty:
            return {}, pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        counts = (
            sub.groupby(["allele", "_label_int_for_sampling"])
            .size()
            .unstack(fill_value=0)
        )

        n_pos = counts[1] if 1 in counts.columns else pd.Series(0, index=counts.index)
        n_neg = counts[0] if 0 in counts.columns else pd.Series(0, index=counts.index)

        summary = pd.DataFrame({
            "allele": counts.index,
            "n_pos": n_pos.astype(int).to_numpy(),
            "n_neg": n_neg.astype(int).to_numpy()
        })
        summary = summary[summary["n_pos"] > 0].reset_index(drop=True)

        if summary.empty:
            return {}, summary

        A = len(summary)
        P = summary["n_pos"].astype(float).to_numpy()
        caps = summary["n_pos"].astype(int).to_numpy()

        target_pos = max(1, int(batch_size / (1.0 + float(neg_ratio))))
        uniform = np.ones(A) / A
        prevalence = P / P.sum()

        denom = np.square(P).sum()
        A_eff = float(A) if denom <= 0 else float((P.sum() ** 2) / denom)
        balance_score = np.clip(A_eff / A, 0.0, 1.0)

        weights = balance_score * uniform + (1.0 - balance_score) * prevalence
        weights /= weights.sum()

        q_pos = self._integer_quota_allocation(
            weights, caps, target_pos, ensure_all_if_possible=False
        )

        q_pos_by_allele = {
            al: int(q) for al, q in zip(summary["allele"], q_pos) if int(q) > 0
        }

        summary["allele_weight"] = weights
        summary["q_pos"] = q_pos
        summary["expected_q_neg_if_available"] = np.floor(q_pos * float(neg_ratio)).astype(int)
        summary["reuse_pressure_pos"] = summary["q_pos"] / summary["n_pos"].clip(lower=1)
        summary["has_negatives"] = summary["n_neg"] > 0

        summary.attrs["target_pos"] = target_pos
        summary.attrs["n_alleles"] = A
        summary.attrs["effective_n_alleles"] = A_eff
        summary.attrs["balance_score"] = float(balance_score)

        return q_pos_by_allele, summary

    # =========================================================
    # Single-domain pooled FT summary
    # =========================================================

    def _compute_pooled_ft_summary(self):
        domain_id = self.update_domain_ids[0]
        cfg = self.ft_cfgs[domain_id]

        sub = self.df[self.df["domain_id"] == domain_id].copy()
        if sub.empty:
            return pd.DataFrame()

        labels = sub["label"].astype(int)
        n_pos_total = int((labels == 1).sum())
        n_neg_total = int((labels == 0).sum())
        n_alleles_pos = int(sub.loc[labels == 1, "allele"].nunique())
        n_alleles_all = int(sub["allele"].nunique())

        pooled_pos = (
            max(1, int(cfg["bs"] / (1.0 + float(cfg["ratio"]))))
            if self.pooled_ft_num_pos is None
            else int(self.pooled_ft_num_pos)
        )

        pooled_neg = int(pooled_pos * float(cfg["ratio"]))

        return pd.DataFrame({
            "mode": ["pooled"],
            "n_pos_total": [n_pos_total],
            "n_neg_total": [n_neg_total],
            "n_alleles_pos": [n_alleles_pos],
            "n_alleles_all": [n_alleles_all],
            "pooled_ft_num_pos": [pooled_pos],
            "expected_pooled_num_neg": [pooled_neg],
            "pos_reuse_pressure": [pooled_pos / max(n_pos_total, 1)]
        })

    # =========================================================
    # Auto-calibration
    # =========================================================

    def _auto_calibrate(self):
        for d, cfg in self.replay_cfgs.items():
            if cfg["bs"] <= 0:
                continue
            q, summary = self._compute_auto_domain_sampling(d, cfg["bs"], cfg["ratio"])
            cfg["q_pos_by_allele"] = q
            cfg["sampling_summary"] = summary

        if len(self.update_domain_ids) == 1 and self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
            d = self.update_domain_ids[0]
            self.ft_cfgs[d]["q_pos_by_allele"] = None
            self.ft_cfgs[d]["sampling_summary"] = self._compute_pooled_ft_summary()
        else:
            for d, cfg in self.ft_cfgs.items():
                q, summary = self._compute_auto_domain_sampling(d, cfg["bs"], cfg["ratio"])
                cfg["q_pos_by_allele"] = q
                cfg["sampling_summary"] = summary

        print("[PepCL_Multi] Replay batch allocation:")
        for d in self.replay_domain_ids:
            label = "MS" if d == 0 else f"domain {d}"
            print(f"  {label}: {self.replay_batch_sizes[d]} / {self.ms_bs}")

        print("[PepCL_Multi] Update batch allocation:")
        for d in self.update_domain_ids:
            print(f"  domain {d}: {self.update_batch_sizes[d]} / {self.ft_bs}")

        print("")

    # =========================================================
    # Loader construction
    # =========================================================

    def _build_multidomain_loader(self, domain_ids, cfgs, batch_sizes, seed, continuous_target=False):
        ds = self._TargetOverrideDataset(
            self.base_dataset, self.df,
            continuous_target=continuous_target,
            continuous_target_col=self.continuous_target_col
        )

        samplers = {}

        for i, d in enumerate(domain_ids):
            cfg = cfgs[d]
            if cfg["bs"] <= 0:
                continue

            sampler = self._InfiniteAlleleSampler(
                self.df,
                domain_id=d,
                n_pos=cfg["n_pos"],
                neg_ratio=cfg["ratio"],
                seed=seed + i,
                q_pos_by_allele=cfg.get("q_pos_by_allele")
            )

            if not sampler.alleles:
                raise ValueError(f"Domain {d} contains no eligible positive alleles.")

            samplers[d] = sampler

        batch_sampler = self._MultiDomainBatchSampler(
            samplers, batch_sizes, seed=seed
        )

        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def _build_single_pooled_ft_loader(self):
        d = self.update_domain_ids[0]
        cfg = self.ft_cfgs[d]

        ds = self._TargetOverrideDataset(
            self.base_dataset, self.df,
            continuous_target=True,
            continuous_target_col=self.continuous_target_col
        )

        sampler = self._InfinitePooledFTSampler(
            self.df,
            domain_id=d,
            pooled_num_pos=cfg.get("pooled_ft_num_pos"),
            neg_ratio=cfg["ratio"],
            batch_size=cfg["bs"],
            seed=self.seed + 10000
        )

        return DataLoader(
            ds,
            batch_size=cfg["bs"],
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    # =========================================================
    # Training
    # =========================================================

    def train(self):
        if self.auto_calibrate:
            self._auto_calibrate()

        need_replay = self.use_replay or self.primary_task_id == 0
        need_ft = self.use_replay or self.primary_task_id == 1

        replay_iter = None
        ft_iter = None

        if need_replay:
            replay_loader = self._build_multidomain_loader(
                self.replay_domain_ids,
                self.replay_cfgs,
                self.replay_batch_sizes,
                seed=self.seed,
                continuous_target=False
            )
            replay_iter = iter(replay_loader)

        if need_ft:
            if (
                len(self.update_domain_ids) == 1
                and self.ft_is_continuous
                and self.continuous_ft_sampling_mode == "pooled"
            ):
                ft_iter = iter(self._build_single_pooled_ft_loader())
            else:
                ft_loader = self._build_multidomain_loader(
                    self.update_domain_ids,
                    self.ft_cfgs,
                    self.update_batch_sizes,
                    seed=self.seed + 10000,
                    continuous_target=self.ft_is_continuous
                )
                ft_iter = iter(ft_loader)

        optimizer = self._get_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(self.steps * self.warmup_ratio),
            self.steps
        )
        scaler = GradScaler()

        metrics = collections.defaultdict(list)
        pbar = tqdm(range(self.steps), desc="PepCL_Multi")

        for step in pbar:
            self.model.train()
            optimizer.zero_grad()

            l_ms_v, l_ft_v, l_dr_v = 0.0, 0.0, 0.0
            loss_ms, loss_ft = None, None

            # -------------------------------------------------
            # Replay stream
            # -------------------------------------------------
            if need_replay:
                try:
                    batch_ms = next(replay_iter)
                    batch_ms = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ms.items()
                    }
                    lbl_ms = batch_ms.pop("label")

                    with autocast(self.device):
                        logits_ms = self.model(batch_ms).view(-1)
                        loss_ms = self.criterion_ms(logits_ms, lbl_ms) * self.ms_lambda

                        if self.dr_lambda > 0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits = self.teacher(batch_ms).view(-1)

                            loss_dr = self._get_dr_loss(logits_ms, t_logits) * self.dr_lambda
                            loss_ms += loss_dr
                            l_dr_v = loss_dr.item()

                    if self.seq_backprop:
                        scaler.scale(loss_ms).backward()

                    l_ms_v = loss_ms.item()

                except (StopIteration, AttributeError):
                    pass

            # -------------------------------------------------
            # FT/update stream
            # -------------------------------------------------
            if need_ft:
                try:
                    batch_ft = next(ft_iter)
                    batch_ft = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ft.items()
                    }
                    lbl_ft = batch_ft.pop("label")

                    with autocast(self.device):
                        logits_ft = self.model(batch_ft).view(-1)
                        loss_ft = self.criterion_ft(logits_ft, lbl_ft) * self.ft_lambda

                    if self.seq_backprop:
                        scaler.scale(loss_ft).backward()

                    l_ft_v = loss_ft.item()

                except (StopIteration, AttributeError):
                    pass

            if not self.seq_backprop:
                total_l = None
                if loss_ms is not None:
                    total_l = loss_ms
                if loss_ft is not None:
                    total_l = loss_ft if total_l is None else total_l + loss_ft
                if total_l is not None:
                    scaler.scale(total_l).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({
                "Replay": f"{l_ms_v:.6f}",
                "FT": f"{l_ft_v:.6f}",
                "DR": f"{l_dr_v:.6f}"
            })

            metrics["ms_loss"].append(l_ms_v)
            metrics["ft_loss"].append(l_ft_v)
            metrics["dr_loss"].append(l_dr_v)

            if self.output_dir and (step + 1) % self.save_every_n_steps == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"pepcl_step_{step + 1}.pt")
                )

        return metrics