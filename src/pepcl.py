
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

class PepCL:
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
        collate_fn = None,

        num_ms_pos: int = 50,
        ms_neg_ratio: float = 1.0,
        num_nonms_pos: int = 50,
        nonms_neg_ratio: float = 1.0,
        ms_batch_size: int = 3072,
        nonms_batch_size: int = 3072,

        auto_calibrate: bool = True,
        calibration_steps: int = 1, # for backward compat

        continuous_target_col: str = "ba_value",
        continuous_ft_sampling_mode: str = "allele_aware",  # "allele_aware" or "pooled"
        pooled_ft_num_pos: int = None,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        update_domain_embeddings: bool = False,

        use_replay: bool = True,
        primary_task_id: int = 1,
        sequential_backprop: bool = True,

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
            raise ValueError(
                "continuous_ft_sampling_mode must be one of: {'allele_aware', 'pooled'}"
            )

        if self.continuous_ft_sampling_mode == "pooled" and not self.ft_is_continuous:
            raise ValueError(
                "continuous_ft_sampling_mode='pooled' is only supported when "
                "nonms_loss_type is continuous, i.e. 'spearman' or 'mse'."
            )

        if self.ft_is_continuous and self.continuous_target_col not in self.df.columns:
            raise ValueError(
                f"Continuous FT requested because nonms_loss_type='{self.nonms_loss_type}', "
                f"but continuous_target_col='{self.continuous_target_col}' is not in train_df."
            )

        self.ms_bs, self.ft_bs = ms_batch_size, nonms_batch_size

        self.auto_calibrate = auto_calibrate
        self.calibration_steps = calibration_steps

        self.ms_cfg = {
            'n_pos': num_ms_pos,
            'ratio': ms_neg_ratio,
            'bs': self.ms_bs,
            'q_pos_by_allele': None,
            'sampling_summary': None,
            'sampling_mode': "allele_aware"
        }
        self.ft_cfg = {
            'n_pos': num_nonms_pos,
            'ratio': nonms_neg_ratio,
            'bs': self.ft_bs,
            'q_pos_by_allele': None,
            'sampling_summary': None,
            'sampling_mode': self.continuous_ft_sampling_mode if self.ft_is_continuous else "allele_aware",
            'pooled_ft_num_pos': self.pooled_ft_num_pos
        }

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
        self.primary_task_id = primary_task_id
        self.seq_backprop = sequential_backprop
        self.update_domain_embeddings = update_domain_embeddings
        self.dr_lambda = dr_lambda
        self.dr_loss_type = dr_loss_type
        self.ms_lambda, self.ft_lambda = ms_lambda, ft_lambda

        self.encoder_lr, self.decoder_lr = encoder_lr, decoder_lr
        self.weight_decay, self.warmup_ratio = weight_decay, warmup_ratio

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        self.teacher = None
        if self.use_replay and self.dr_lambda > 0:
            self.teacher = copy.deepcopy(model).to(device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        torch.manual_seed(seed)
        np.random.seed(seed)

    class _LogSmoothAPScore(nn.Module):
        def __init__(self, alpha=10.0, epsilon=1e-8):
            super().__init__()
            self.alpha, self.epsilon = alpha, epsilon

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
        """
        Differentiable rank based on pairwise sigmoid.
        r_i = 1 + Σ_j σ((x_j - x_i) / τ)
        τ -> 0 gives hard ranks.
        x : (..., L)
        returns (..., L) in [1, L]
        """
        x_col = x.unsqueeze(-1)
        x_row = x.unsqueeze(-2)
        P = torch.sigmoid((x_row - x_col) / tau)
        return 1 + P.sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(self, tau: float = 0.1):
            super().__init__()
            self.tau = tau

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            # ensure 2-D
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            r_pred = PepCL._soft_rank(pred, tau=self.tau)
            r_target = PepCL._soft_rank(target, tau=1e-4)

            r_pred = r_pred - r_pred.mean(-1, keepdim=True)
            r_target = r_target - r_target.mean(-1, keepdim=True)

            num = (r_pred * r_target).sum(-1)
            den = torch.norm(r_pred, dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
            rho = num / den

            return (1 - rho).mean()

    class _TargetOverrideDataset(Dataset):
        """
        Wraps the parent PeptideMHCDataset so a stream can override:
            - domain_id
            - label target

        For classification streams:
            label remains the binary label from the parent dataset / dataframe.

        For continuous FT streams:
            label becomes peptide_{continuous_target_col}, e.g. peptide_ba_value,
            or falls back to df[continuous_target_col].
        """
        def __init__(
            self,
            parent_ds,
            df,
            domain_id,
            continuous_target: bool = False,
            continuous_target_col: str = "ba_value"
        ):
            self.parent_ds = parent_ds
            self.df = df
            self.domain_id = domain_id
            self.continuous_target = continuous_target
            self.continuous_target_col = continuous_target_col
            self.item_continuous_key = f"peptide_{continuous_target_col}"

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = self.parent_ds[idx]
            item = dict(item)

            item['domain_id'] = torch.tensor(self.domain_id, dtype=torch.long)

            if self.continuous_target:
                if self.item_continuous_key in item:
                    target = item[self.item_continuous_key]
                    if isinstance(target, torch.Tensor):
                        item['label'] = target.float()
                    else:
                        item['label'] = torch.tensor(target, dtype=torch.float)
                else:
                    target = self.df.loc[idx, self.continuous_target_col]
                    item['label'] = torch.tensor(target, dtype=torch.float)

            return item

    class _InfiniteAlleleSampler(Sampler):
        """
        Infinite allele-aware sampler.

        Manual mode:
            Uses scalar n_pos per allele per sampler cycle.

        Auto mode:
            Uses q_pos_by_allele, a dict mapping allele -> positive quota per sampler cycle.

        Negative behavior:
            neg_ratio = 1.0 means sample equal number of negatives per selected positives.
            neg_ratio = 0.0 means sample no negatives.
            If an allele has no negatives, this is allowed and no negatives are sampled for that allele.
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
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.n_pos = n_pos
            self.neg_ratio = float(neg_ratio)
            self.q_pos_by_allele = q_pos_by_allele

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            self.pools = {0: {}, 1: {}}

            subset = df[df['domain_id'] == domain_id]
            if subset.empty:
                self.alleles = []
                return

            for (lbl, allele), group in subset.groupby(['label', 'allele']):
                lbl_int = int(lbl)
                if lbl_int not in self.pools:
                    continue
                idx = group.index.to_numpy()
                self.rng.shuffle(idx)
                self.pools[lbl_int][allele] = idx

            positive_alleles = sorted(list(self.pools[1].keys()))

            if self.q_pos_by_allele is not None:
                self.q_pos_by_allele = {
                    al: int(q)
                    for al, q in self.q_pos_by_allele.items()
                    if int(q) > 0
                }
                self.alleles = [
                    al for al in positive_alleles
                    if self.q_pos_by_allele.get(al, 0) > 0
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

                    if self.q_pos_by_allele is not None:
                        requested_pos = int(self.q_pos_by_allele.get(al, 0))
                    else:
                        requested_pos = int(self.n_pos)

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
                            sel_neg = self.rng.choice(neg_data, size=n_take, replace=False)
                            indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    class _InfinitePooledFTSampler(Sampler):
        """
        Infinite global pooled sampler for continuous FT/BA mode.

        This mimics the important BA-side behavior of the older
        ContinuousRecyclingMTLDataLoaderFactory:

            - collect all domain 1 positives globally across alleles
            - collect all domain 1 negatives globally across alleles
            - sample pooled_ft_num_pos positives globally
            - sample negatives globally according to neg_ratio
            - yield shuffled indices forever

        The dataset wrapper handles using ba_value as the label target.
        """
        def __init__(
            self,
            df,
            domain_id,
            pooled_num_pos,
            neg_ratio,
            batch_size,
            seed
        ):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.neg_ratio = float(neg_ratio)
            self.batch_size = int(batch_size)

            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

            subset = df[df['domain_id'] == domain_id].copy()
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                self.pooled_num_pos = 0
                return

            label_int = subset['label'].astype(int)

            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()

            if pooled_num_pos is None:
                # Default: one batch/cycle worth of positives implied by batch size and neg ratio.
                pooled_num_pos = int(self.batch_size / (1.0 + self.neg_ratio))
                pooled_num_pos = max(1, pooled_num_pos)

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
                        sel_neg = self.rng.choice(self.neg_pool, size=n_req, replace=replace)
                        indices.extend(sel_neg)

                if len(indices) == 0:
                    return

                self.rng.shuffle(indices)
                for i in indices:
                    yield i

    def _get_criterion(self, t, alpha):
        t = str(t).lower()

        if t == "logsmoothap":
            return self._LogSmoothAPScore(alpha=alpha)

        if t == "spearman":
            return self._SoftSpearmanLoss(tau=alpha)

        if t == "mse":
            return nn.MSELoss()

        if t in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
            return nn.BCEWithLogitsLoss()

        return nn.BCEWithLogitsLoss()

    def _get_dr_loss(self, student_logits, teacher_logits):
        if self.dr_loss_type == "mse":
            return F.mse_loss(student_logits, teacher_logits)
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
            [
                {'params': e, 'lr': self.encoder_lr},
                {'params': d, 'lr': self.decoder_lr}
            ],
            weight_decay=self.weight_decay
        )

    @staticmethod
    def _integer_quota_allocation(weights, caps, total, ensure_all_if_possible=True):
        """
        Allocate an integer total across alleles according to weights, respecting caps.
        """
        weights = np.asarray(weights, dtype=float)
        caps = np.asarray(caps, dtype=int)
        total = int(total)

        q = np.zeros_like(caps, dtype=int)

        valid = caps > 0
        if total <= 0 or not np.any(valid):
            return q

        max_total = int(min(total, caps.sum()))
        if max_total <= 0:
            return q

        w = np.zeros_like(weights, dtype=float)
        if weights[valid].sum() > 0:
            w[valid] = weights[valid] / weights[valid].sum()
        else:
            w[valid] = 1.0 / valid.sum()

        remaining_total = max_total

        if ensure_all_if_possible and remaining_total >= valid.sum():
            q[valid] = 1
            remaining_total -= int(valid.sum())

        if remaining_total <= 0:
            return q

        while remaining_total > 0:
            capacity = caps - q
            can_receive = capacity > 0

            if not np.any(can_receive):
                break

            w_active = np.zeros_like(w, dtype=float)
            if w[can_receive].sum() > 0:
                w_active[can_receive] = w[can_receive] / w[can_receive].sum()
            else:
                w_active[can_receive] = 1.0 / can_receive.sum()

            raw = remaining_total * w_active
            add = np.floor(raw).astype(int)
            add = np.minimum(add, capacity)

            if add.sum() > 0:
                q += add
                remaining_total -= int(add.sum())
                if remaining_total <= 0:
                    break

            capacity = caps - q
            can_receive = capacity > 0
            if not np.any(can_receive):
                break

            frac = raw - np.floor(raw)
            frac[~can_receive] = -1.0

            order = np.argsort(-frac)
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

    def _compute_auto_domain_sampling(self, domain_id, batch_size, neg_ratio):
        """
        Data-adaptive allele-aware sampling calibration.

        This does not look at model loss and does not change gradient logic.
        It only computes per-allele positive quotas for batch construction.
        """
        if neg_ratio < 0:
            raise ValueError("neg_ratio must be >= 0. Use 1.0 for equal negatives, 0.0 for no negatives.")

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

        target_pos = int(batch_size / (1.0 + float(neg_ratio)))
        target_pos = max(1, target_pos)

        uniform = np.ones(A, dtype=float) / A
        prevalence = P / P.sum()

        denom = np.square(P).sum()
        if denom <= 0:
            A_eff = float(A)
        else:
            A_eff = float((P.sum() ** 2) / denom)

        balance_score = float(A_eff / A)
        balance_score = max(0.0, min(1.0, balance_score))

        weights = balance_score * uniform + (1.0 - balance_score) * prevalence
        weights = weights / weights.sum()

        q_pos = self._integer_quota_allocation(
            weights=weights,
            caps=caps,
            total=target_pos,
            ensure_all_if_possible=False
        )

        q_pos_by_allele = {
            al: int(q)
            for al, q in zip(summary["allele"].tolist(), q_pos)
            if int(q) > 0
        }

        summary["allele_weight"] = weights
        summary["q_pos"] = q_pos.astype(int)
        summary["expected_q_neg_if_available"] = np.floor(
            summary["q_pos"].to_numpy() * float(neg_ratio)
        ).astype(int)
        summary["reuse_pressure_pos"] = summary["q_pos"] / summary["n_pos"].clip(lower=1)

        summary["has_negatives"] = summary["n_neg"] > 0

        summary.attrs["target_pos"] = target_pos
        summary.attrs["n_alleles"] = A
        summary.attrs["effective_n_alleles"] = A_eff
        summary.attrs["balance_score"] = balance_score

        return q_pos_by_allele, summary

    def _compute_pooled_ft_summary(self):
        """
        Diagnostic summary for pooled continuous FT mode.
        This does not affect sampling except determining the default pooled_ft_num_pos if None.
        """
        sub = self.df[self.df["domain_id"] == 1].copy()

        if sub.empty:
            return pd.DataFrame()

        sub["_label_int_for_sampling"] = sub["label"].astype(int)

        n_pos_total = int((sub["_label_int_for_sampling"] == 1).sum())
        n_neg_total = int((sub["_label_int_for_sampling"] == 0).sum())
        n_alleles_pos = int(sub.loc[sub["_label_int_for_sampling"] == 1, "allele"].nunique())
        n_alleles_all = int(sub["allele"].nunique())

        if self.pooled_ft_num_pos is None:
            pooled_pos = int(self.ft_cfg["bs"] / (1.0 + float(self.ft_cfg["ratio"])))
            pooled_pos = max(1, pooled_pos)
        else:
            pooled_pos = int(self.pooled_ft_num_pos)

        pooled_neg = int(pooled_pos * float(self.ft_cfg["ratio"]))

        summary = pd.DataFrame({
            "mode": ["pooled"],
            "n_pos_total": [n_pos_total],
            "n_neg_total": [n_neg_total],
            "n_alleles_pos": [n_alleles_pos],
            "n_alleles_all": [n_alleles_all],
            "pooled_ft_num_pos": [pooled_pos],
            "expected_pooled_num_neg": [pooled_neg],
            "pos_reuse_pressure": [pooled_pos / max(n_pos_total, 1)]
        })

        summary.attrs["pooled_ft_num_pos"] = pooled_pos
        summary.attrs["expected_pooled_num_neg"] = pooled_neg

        return summary

    def _auto_calibrate(self):
        """
        Auto-calibration only controls sampling parameters.

        For MS:
            always allele-aware.

        For FT:
            - allele_aware: compute per-allele positive quotas.
            - pooled: do not compute per-allele quotas; print pooled diagnostics.
        """
        ms_q, ms_summary = self._compute_auto_domain_sampling(
            domain_id=0,
            batch_size=self.ms_cfg["bs"],
            neg_ratio=self.ms_cfg["ratio"]
        )

        self.ms_cfg["q_pos_by_allele"] = ms_q
        self.ms_cfg["sampling_summary"] = ms_summary

        if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
            ft_summary = self._compute_pooled_ft_summary()
            self.ft_cfg["q_pos_by_allele"] = None
            self.ft_cfg["sampling_summary"] = ft_summary
        else:
            ft_q, ft_summary = self._compute_auto_domain_sampling(
                domain_id=1,
                batch_size=self.ft_cfg["bs"],
                neg_ratio=self.ft_cfg["ratio"]
            )
            self.ft_cfg["q_pos_by_allele"] = ft_q
            self.ft_cfg["sampling_summary"] = ft_summary

        def _summary_line(name, summary, cfg):
            if summary is None or summary.empty:
                return f"  {name}: no eligible positive alleles"

            if "mode" in summary.columns and summary["mode"].iloc[0] == "pooled":
                row = summary.iloc[0]
                return (
                    f"  {name}: pooled continuous FT, "
                    f"alleles_pos={int(row['n_alleles_pos'])}, alleles_all={int(row['n_alleles_all'])}, "
                    f"pos_pool={int(row['n_pos_total'])}, neg_pool={int(row['n_neg_total'])}, "
                    f"pooled_pos/cycle={int(row['pooled_ft_num_pos'])}, "
                    f"expected_neg/cycle={int(row['expected_pooled_num_neg'])}, "
                    f"pos_reuse={float(row['pos_reuse_pressure']):.4f}, "
                    f"neg_ratio={cfg['ratio']}"
                )

            # Allele-aware summary
            target_pos = summary.attrs.get("target_pos", None)
            A = summary.attrs.get("n_alleles", len(summary))
            A_eff = summary.attrs.get("effective_n_alleles", np.nan)
            balance_score = summary.attrs.get("balance_score", np.nan)

            q_nonzero = summary.loc[summary["q_pos"] > 0, "q_pos"]
            reuse = summary.loc[summary["q_pos"] > 0, "reuse_pressure_pos"]

            q_med = float(q_nonzero.median()) if len(q_nonzero) else 0.0
            q_min = int(q_nonzero.min()) if len(q_nonzero) else 0
            q_max = int(q_nonzero.max()) if len(q_nonzero) else 0

            reuse_med = float(reuse.median()) if len(reuse) else 0.0
            reuse_max = float(reuse.max()) if len(reuse) else 0.0

            return (
                f"  {name}: alleles={A}, A_eff={A_eff:.2f}, balance={balance_score:.3f}, "
                f"target_pos/cycle={target_pos}, q_pos median={q_med:.1f} "
                f"[{q_min}, {q_max}], reuse median={reuse_med:.4f}, reuse max={reuse_max:.4f}, "
                f"neg_ratio={cfg['ratio']}"
            )

        print("[PepCL] Auto-calibrated sampling:")
        print(_summary_line("MS", self.ms_cfg["sampling_summary"], self.ms_cfg))
        print(_summary_line("FT", self.ft_cfg["sampling_summary"], self.ft_cfg))
        print("")

    def train(self):
        if self.auto_calibrate:
            self._auto_calibrate()
        else:
            if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
                self.ft_cfg["sampling_summary"] = self._compute_pooled_ft_summary()

            print(
                f"[PepCL] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, FT neg_ratio={self.ft_cfg['ratio']} | "
                f"FT continuous={self.ft_is_continuous}, "
                f"FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )

        def _build_loader(domain_id, cfg, seed):
            continuous_target = (
                domain_id == 1 and
                self.ft_is_continuous
            )

            ds = self._TargetOverrideDataset(
                parent_ds=self.base_dataset,
                df=self.df,
                domain_id=domain_id,
                continuous_target=continuous_target,
                continuous_target_col=self.continuous_target_col
            )

            if (
                domain_id == 1 and
                self.ft_is_continuous and
                cfg.get("sampling_mode", "allele_aware") == "pooled"
            ):
                sampler = self._InfinitePooledFTSampler(
                    self.df,
                    domain_id=domain_id,
                    pooled_num_pos=cfg.get("pooled_ft_num_pos", None),
                    neg_ratio=cfg["ratio"],
                    batch_size=cfg["bs"],
                    seed=seed
                )
            else:
                sampler = self._InfiniteAlleleSampler(
                    self.df,
                    domain_id,
                    cfg['n_pos'],
                    cfg['ratio'],
                    seed,
                    q_pos_by_allele=cfg.get("q_pos_by_allele", None)
                )

            return DataLoader(
                ds,
                batch_size=cfg['bs'],
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=(self.num_workers > 0)
            )

        ms_iter = iter(_build_loader(0, self.ms_cfg, self.seed))
        ft_iter = iter(_build_loader(1, self.ft_cfg, self.seed + 1))

        optimizer = self._get_optimizer()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(self.steps * self.warmup_ratio),
            self.steps
        )
        scaler = GradScaler()

        metrics = collections.defaultdict(list)
        pbar = tqdm(range(self.steps), desc="PepCL")

        for step in pbar:
            self.model.train()
            optimizer.zero_grad()

            l_ms_v, l_ft_v, l_dr_v = 0.0, 0.0, 0.0

            # Safety initialization for non-sequential mode.
            # This prevents stale tensors if one stream is unavailable in a given step.
            loss_ms = None
            loss_ft = None

            # Domain 0 / Replay
            if self.use_replay or self.primary_task_id == 0:
                try:
                    batch_ms = next(ms_iter)
                    batch_ms = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ms.items()
                    }
                    lbl_ms = batch_ms.pop('label')

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

            if self.use_replay or self.primary_task_id == 1:
                try:
                    batch_ft = next(ft_iter)
                    batch_ft = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ft.items()
                    }
                    lbl_ft = batch_ft.pop('label')

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
                'MS': f"{l_ms_v:.6f}",
                'FT': f"{l_ft_v:.6f}",
                'DR': f"{l_dr_v:.6f}"
            })

            metrics['ms_loss'].append(l_ms_v)
            metrics['ft_loss'].append(l_ft_v)
            metrics['dr_loss'].append(l_dr_v)

            if self.output_dir and (step + 1) % self.save_every_n_steps == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"pepcl_step_{step+1}.pt")
                )

        return metrics
