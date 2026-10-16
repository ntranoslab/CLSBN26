# logged version of pepcl

import collections
import copy
import os
import io
import time
import warnings
import socket
import sys
import platform
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

try:
    import psutil
except ImportError:
    psutil = None

from .model_utils import PeptideMHCDataset

class PepCL_Logging:
    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps = None,
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
        allele_aware_sampling: bool = True,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
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
        if save_every_n_steps is not None:
            if isinstance(save_every_n_steps, (int, np.integer)):
                if int(save_every_n_steps) <= 0:
                    raise ValueError("save_every_n_steps must be > 0 when given as an integer.")
                self.save_every_n_steps = int(save_every_n_steps)
            elif isinstance(save_every_n_steps, list):
                if not all(isinstance(x, (int, np.integer)) for x in save_every_n_steps):
                    raise ValueError("All values in save_every_n_steps must be integers.")
                if any(int(x) <= 0 for x in save_every_n_steps):
                    raise ValueError("All save steps must be > 0.")
                self.save_every_n_steps = {int(x) for x in save_every_n_steps}
            else:
                raise TypeError("save_every_n_steps must be None, a positive integer, or a list of positive integers.")
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
        self.allele_aware_sampling = bool(allele_aware_sampling)

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
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        if self.lr_scheduler_type not in {"linear", "warmup_constant", "constant"}:
            raise ValueError(
                "lr_scheduler_type must be one of: {'linear', 'warmup_constant', 'constant'}"
            )

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        self.teacher = None
        if self.dr_lambda > 0:
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

            r_pred = PepCL_Logging._soft_rank(pred, tau=self.tau)
            r_target = PepCL_Logging._soft_rank(target, tau=1e-4)

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

    class _InfiniteBalancedSampler(Sampler):
        """
        Infinite global 1:1 positive/negative sampler within one domain.

        This is the naive non-allele-aware ablation: each DataLoader batch is
        constructed from approximately half positives and half negatives, with
        no allele stratification and no positive-count hyperparameter.
        """
        def __init__(self, df, domain_id, batch_size, seed):
            self.rng = np.random.default_rng(seed)
            self.batch_size = int(batch_size)

            subset = df[df['domain_id'] == domain_id]
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                return

            label_int = subset['label'].astype(int)
            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        def __iter__(self):
            if len(self.pos_pool) == 0 or len(self.neg_pool) == 0:
                return

            n_pos = self.batch_size // 2
            n_neg = self.batch_size - n_pos

            while True:
                sel_pos = self.rng.choice(self.pos_pool, size=n_pos, replace=len(self.pos_pool) < n_pos)
                sel_neg = self.rng.choice(self.neg_pool, size=n_neg, replace=len(self.neg_pool) < n_neg)
                indices = np.concatenate([sel_pos, sel_neg])
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
        if not self.allele_aware_sampling:
            print(
                "[PepCL] Naive balanced sampling: allele-aware sampling disabled; "
                "each active stream samples 1:1 positives/negatives globally within the stream. "
                "num_*_pos and *_neg_ratio are ignored for sampling in this mode.\n"
            )
        elif self.auto_calibrate:
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

            if not self.allele_aware_sampling:
                sampler = self._InfiniteBalancedSampler(
                    self.df, domain_id=domain_id, batch_size=cfg['bs'], seed=seed
                )
            elif (
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
        warmup_steps = int(self.steps * self.warmup_ratio)
        if self.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.steps)
        elif self.lr_scheduler_type == "warmup_constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
        else:
            scheduler = get_constant_schedule(optimizer)
        scaler = GradScaler()

        metrics = collections.defaultdict(list)
        model_checkpoints = {} if self.save_every_n_steps is not None and self.output_dir is None else None
        pbar = tqdm(range(self.steps), desc="PepCL_Logging")

        for step in pbar:
            self.model.train()
            optimizer.zero_grad()

            l_ms_v, l_ft_v, l_dr_v = 0.0, 0.0, 0.0

            # Safety initialization for non-sequential mode.
            # This prevents stale tensors if one stream is unavailable in a given step.
            loss_ms = None
            loss_ft = None

            # Domain 0 / Replay and/or drift regularization
            compute_ms_classification = self.use_replay or self.primary_task_id == 0
            if compute_ms_classification or self.dr_lambda > 0:
                try:
                    batch_ms = next(ms_iter)
                    batch_ms = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch_ms.items()
                    }
                    lbl_ms = batch_ms.pop('label')

                    with autocast(self.device):
                        logits_ms = self.model(batch_ms).view(-1)
                        loss_ms_cls = None
                        loss_dr = None

                        if compute_ms_classification:
                            loss_ms_cls = self.criterion_ms(logits_ms, lbl_ms) * self.ms_lambda

                        if self.dr_lambda > 0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits = self.teacher(batch_ms).view(-1)
                            loss_dr = self._get_dr_loss(logits_ms, t_logits) * self.dr_lambda
                            l_dr_v = loss_dr.item()

                        if loss_ms_cls is not None and loss_dr is not None:
                            loss_ms = loss_ms_cls + loss_dr
                        elif loss_ms_cls is not None:
                            loss_ms = loss_ms_cls
                        else:
                            loss_ms = loss_dr

                    if self.seq_backprop and loss_ms is not None:
                        scaler.scale(loss_ms).backward()

                    if compute_ms_classification:
                        # Preserve legacy metric behavior: with replay+DR, ms_loss is the combined MS+DR loss.
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

            completed_step = step + 1
            if self.save_every_n_steps is None:
                should_save = False
            elif isinstance(self.save_every_n_steps, int):
                should_save = completed_step % self.save_every_n_steps == 0
            else:
                should_save = completed_step in self.save_every_n_steps

            if should_save:
                if self.output_dir is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir, f"pepcl_step_{completed_step}.pt")
                    )
                else:
                    model_checkpoints[completed_step] = copy.deepcopy(self.model).cpu().eval()

        if model_checkpoints is not None:
            return metrics, model_checkpoints
        return metrics

try:
    import psutil
except ImportError:
    psutil = None

class PepCL_Logging_Ext:
    """
    PepCL_Logging with two engineering additions only:
      1) a compact minimal-data path that reconstructs the exact model-facing peptide/MHC inputs on demand;
      2) optional low-overhead resource/compute logging.

    Core PepCL behavior is otherwise preserved: sampling ablations, continuous FT modes,
    LR scheduler choices, replay/DR behavior, sequential/non-sequential backprop, and
    checkpoint capture all remain available.

    Minimal train_df columns
    ------------------------
    Classification:
        seq, allele, label, domain_id
    Continuous FT:
        seq, allele, label, domain_id, <continuous_target_col>

    Peptide construction is exactly:
        10 pad flank tokens + raw peptide + right pad to peptide length 14 + 10 pad flank tokens
    giving exactly 34 peptide-side tokens before collation.

    MHC pseudosequences are looked up from mhc_seq_dict and tokenized once per unique allele.
    """

    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,
        mhc_seq_dict: dict,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps=None,
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
        allele_aware_sampling: bool = True,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
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
        dr_loss_type: str = "mse",

        # Compact-data / loader controls. Defaults preserve current model-facing layout.
        compact_dataframe: bool = True,
        compact_dtypes: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,

        # Optional logging. Logging does not change training behavior.
        enable_logging: bool = True,
        telemetry_every_n_steps: int = 10,
        enable_nvml_sampling: bool = True,
        measure_serialized_model_size: bool = False,

    ):
        self._object_t0 = time.perf_counter()
        # Device must be assigned before NVML initialization; otherwise a requested cuda:1
        # could incorrectly fall back to the process current device (often cuda:0).
        self.device = torch.device(device)
        self.enable_logging = bool(enable_logging)
        self.telemetry_every_n_steps = max(1, int(telemetry_every_n_steps))
        self.enable_nvml_sampling = bool(enable_nvml_sampling)
        self.measure_serialized_model_size = bool(measure_serialized_model_size)
        self._process = psutil.Process(os.getpid()) if (self.enable_logging and psutil is not None) else None
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        self._nvml = self._init_nvml() if self.enable_logging and self.enable_nvml_sampling else None

        self.logs = {
            "config": {}, "data": {}, "model": {}, "sampling": {},
            "timing": {}, "compute": collections.defaultdict(float),
            "environment": {}, "data_loader": {}, "memory_baseline": {},
            "memory_snapshots": [], "step_telemetry": [],
            "cuda_allocator_stats_end": {}, "notes": [],
        }
        self._snapshot("init_enter", detailed_memory=True)
        if self.logs["memory_snapshots"]:
            self.logs["memory_baseline"] = copy.deepcopy(self.logs["memory_snapshots"][-1])

        self.tokenizer = tokenizer
        self.mhc_seq_dict = mhc_seq_dict
        self.steps = int(steps)
        self.output_dir = output_dir
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.user_collate_fn = collate_fn
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.compact_dataframe = bool(compact_dataframe)
        self.compact_dtypes = bool(compact_dtypes)

        # Fixed model-input geometry from the existing pipeline.
        self.n_flank_len = 10
        self.peptide_core_max_len = 14
        self.c_flank_len = 10
        self.model_peptide_len = 34
        self.model_mhc_len = 34

        # Save configuration parsing exactly as PepCL_Logging.
        self.save_every_n_steps = save_every_n_steps
        if save_every_n_steps is not None:
            if isinstance(save_every_n_steps, (int, np.integer)):
                if int(save_every_n_steps) <= 0:
                    raise ValueError("save_every_n_steps must be > 0 when given as an integer.")
                self.save_every_n_steps = int(save_every_n_steps)
            elif isinstance(save_every_n_steps, list):
                if not all(isinstance(x, (int, np.integer)) for x in save_every_n_steps):
                    raise ValueError("All values in save_every_n_steps must be integers.")
                if any(int(x) <= 0 for x in save_every_n_steps):
                    raise ValueError("All save steps must be > 0.")
                self.save_every_n_steps = {int(x) for x in save_every_n_steps}
            else:
                raise TypeError("save_every_n_steps must be None, a positive integer, or a list of positive integers.")

        # Store method configuration before deciding active streams.
        self.ms_loss_type = str(ms_loss_type).lower()
        self.nonms_loss_type = str(nonms_loss_type).lower()
        self.continuous_ft_loss_types = {"spearman", "mse"}
        self.ft_is_continuous = self.nonms_loss_type in self.continuous_ft_loss_types
        self.continuous_target_col = continuous_target_col
        self.continuous_ft_sampling_mode = continuous_ft_sampling_mode
        self.pooled_ft_num_pos = pooled_ft_num_pos
        self.allele_aware_sampling = bool(allele_aware_sampling)
        self.use_replay = bool(use_replay)
        self.primary_task_id = int(primary_task_id)
        self.seq_backprop = bool(sequential_backprop)
        self.update_domain_embeddings = bool(update_domain_embeddings)
        self.dr_lambda = float(dr_lambda)
        self.dr_loss_type = dr_loss_type
        self.ms_lambda, self.ft_lambda = float(ms_lambda), float(ft_lambda)
        self.encoder_lr, self.decoder_lr = float(encoder_lr), float(decoder_lr)
        self.weight_decay, self.warmup_ratio = float(weight_decay), float(warmup_ratio)
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        self.auto_calibrate = bool(auto_calibrate)
        self.calibration_steps = calibration_steps
        self.ms_bs, self.ft_bs = int(ms_batch_size), int(nonms_batch_size)

        if self.continuous_ft_sampling_mode not in {"allele_aware", "pooled"}:
            raise ValueError("continuous_ft_sampling_mode must be one of: {'allele_aware', 'pooled'}")
        if self.continuous_ft_sampling_mode == "pooled" and not self.ft_is_continuous:
            raise ValueError("continuous_ft_sampling_mode='pooled' is only supported for 'spearman' or 'mse'.")
        if self.lr_scheduler_type not in {"linear", "warmup_constant", "constant"}:
            raise ValueError("lr_scheduler_type must be one of: {'linear', 'warmup_constant', 'constant'}")
        if self.primary_task_id not in {0, 1}:
            raise ValueError("primary_task_id must be 0 (MS/replay stream) or 1 (FT/update stream).")

        self.compute_ms_classification = self.use_replay or self.primary_task_id == 0
        self.need_ms_stream = self.compute_ms_classification or self.dr_lambda > 0
        self.need_ft_stream = self.use_replay or self.primary_task_id == 1

        # Data compaction / validation.
        t = time.perf_counter()
        self.logs["data"]["input_dataframe_rows"] = int(len(train_df))
        self.logs["data"]["input_dataframe_memory_bytes"] = self._df_bytes(train_df)
        self.df = self._prepare_compact_dataframe(train_df)
        self.logs["data"]["compact_dataframe_rows"] = int(len(self.df))
        self.logs["data"]["compact_dataframe_memory_bytes"] = self._df_bytes(self.df)
        self.logs["data"]["compact_dataframe_columns"] = list(self.df.columns)
        self.logs["data"]["domain_counts"] = {int(k): int(v) for k, v in self.df["domain_id"].value_counts().sort_index().items()}
        self.logs["data"]["positive_counts_by_domain"] = {
            int(k): int(v) for k, v in self.df.loc[self.df["label"].astype(int) == 1, "domain_id"].value_counts().sort_index().items()
        }
        self.logs["timing"]["data_compaction_seconds"] = time.perf_counter() - t
        self._snapshot("after_data_compaction")

        # Required domain checks. FT-only is allowed to contain only domain 1.
        domains = set(self.df["domain_id"].astype(int).unique().tolist())
        if self.need_ms_stream and 0 not in domains:
            raise ValueError("Domain 0 is required because replay/MS classification or drift regularization is active.")
        if self.need_ft_stream and 1 not in domains:
            raise ValueError("Domain 1 is required because the FT/update stream is active.")
        if self.ft_is_continuous and self.continuous_target_col not in self.df.columns:
            raise ValueError(
                f"Continuous FT requested because nonms_loss_type='{self.nonms_loss_type}', but "
                f"continuous_target_col='{self.continuous_target_col}' is not in train_df."
            )

        self.ms_cfg = {
            "n_pos": num_ms_pos, "ratio": ms_neg_ratio, "bs": self.ms_bs,
            "q_pos_by_allele": None, "sampling_summary": None, "sampling_mode": "allele_aware",
        }
        self.ft_cfg = {
            "n_pos": num_nonms_pos, "ratio": nonms_neg_ratio, "bs": self.ft_bs,
            "q_pos_by_allele": None, "sampling_summary": None,
            "sampling_mode": self.continuous_ft_sampling_mode if self.ft_is_continuous else "allele_aware",
            "pooled_ft_num_pos": self.pooled_ft_num_pos,
        }

        # Minimal dataset creation; MHC sequences are tokenized once per unique allele.
        t = time.perf_counter()
        self.base_dataset = self._MinimalPeptideMHCDataset(
            self.df, self.tokenizer, self.mhc_seq_dict,
            continuous_target_col=self.continuous_target_col if self.ft_is_continuous else None,
            n_flank_len=self.n_flank_len, peptide_core_max_len=self.peptide_core_max_len,
            c_flank_len=self.c_flank_len, mhc_max_len=self.model_mhc_len,
        )
        self.collate_fn = self.user_collate_fn if self.user_collate_fn is not None else self._minimal_collate_fn
        self.logs["timing"]["minimal_dataset_init_seconds"] = time.perf_counter() - t
        self.logs["data"]["unique_alleles"] = int(self.df["allele"].nunique())
        self._snapshot("after_minimal_dataset_init")

        # Move live model to requested device and record model footprint.
        t = time.perf_counter()
        self.model = model.to(self.device)
        self.logs["timing"]["model_to_device_seconds"] = time.perf_counter() - t
        self.logs["model"].update(self._model_stats(self.model, prefix="student"))
        self._snapshot("after_model_to_device")

        self.criterion_ms = self._get_criterion(self.ms_loss_type, ms_loss_alpha)
        self.criterion_ft = self._get_criterion(self.nonms_loss_type, nonms_loss_alpha)

        # Frozen teacher behavior is unchanged: when DR is active, clone the incoming current model.
        self.teacher = None
        if self.dr_lambda > 0:
            t = time.perf_counter()
            self._snapshot("before_teacher_copy")
            self.teacher = copy.deepcopy(model).to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.logs["timing"]["teacher_copy_seconds"] = time.perf_counter() - t
            self.logs["model"].update(self._model_stats(self.teacher, prefix="teacher"))
            self._snapshot("after_teacher_copy")
        else:
            self.logs["timing"]["teacher_copy_seconds"] = 0.0

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.logs["config"] = {
            "steps": self.steps, "seed": self.seed, "num_workers": self.num_workers,
            "use_replay": self.use_replay, "primary_task_id": self.primary_task_id,
            "dr_lambda": self.dr_lambda, "sequential_backprop": self.seq_backprop,
            "allele_aware_sampling": self.allele_aware_sampling, "auto_calibrate": self.auto_calibrate,
            "continuous_ft_sampling_mode": self.continuous_ft_sampling_mode,
            "lr_scheduler_type": self.lr_scheduler_type, "warmup_ratio": self.warmup_ratio,
            "ms_batch_size": self.ms_bs, "nonms_batch_size": self.ft_bs,
            "compact_dataframe": self.compact_dataframe, "compact_dtypes": self.compact_dtypes,
            "pin_memory": self.pin_memory, "persistent_workers": self.persistent_workers,
            "telemetry_every_n_steps": self.telemetry_every_n_steps,
        }
        self.logs["environment"] = self._environment_stats()
        self.logs["timing"]["constructor_total_seconds"] = time.perf_counter() - self._object_t0
        self._snapshot("init_exit", detailed_memory=True)

    # =====================================================================
    # Compact dataset / exact input reconstruction
    # =====================================================================

    class _MinimalPeptideMHCDataset(Dataset):
        def __init__(
            self, df, tokenizer, mhc_seq_dict, continuous_target_col=None,
            n_flank_len=10, peptide_core_max_len=14, c_flank_len=10, mhc_max_len=34,
        ):
            self.tokenizer = tokenizer
            self.mhc_seq_dict = mhc_seq_dict
            self.continuous_target_col = continuous_target_col
            self.n_flank_len = int(n_flank_len)
            self.peptide_core_max_len = int(peptide_core_max_len)
            self.c_flank_len = int(c_flank_len)
            self.mhc_max_len = int(mhc_max_len)
            self.pad_idx = int(tokenizer.pad_idx)

            # Avoid duplicating a large dataframe inside the Dataset. These arrays either view or
            # cheaply reference the compact columns. Allele categorical codes are tiny.
            self.seqs = df["seq"].to_numpy(copy=False)
            if isinstance(df["allele"].dtype, pd.CategoricalDtype):
                self.allele_codes = df["allele"].cat.codes.to_numpy(copy=False)
                self.allele_categories = np.asarray(df["allele"].cat.categories.astype(str), dtype=object)
                self.alleles = None
            else:
                self.allele_codes = None
                self.allele_categories = None
                self.alleles = df["allele"].to_numpy(copy=False)
            self.labels = df["label"].to_numpy(copy=False)
            self.domain_ids = df["domain_id"].to_numpy(copy=False)
            self.continuous_values = (
                df[continuous_target_col].to_numpy(copy=False)
                if continuous_target_col is not None and continuous_target_col in df.columns else None
            )

            # Pretokenize only the tiny unique MHC dictionary actually referenced by the dataframe.
            unique_fields = df["allele"].astype(str).unique().tolist()
            unique_alleles = set()
            for field in unique_fields:
                unique_alleles.update(str(field).split("_"))
            missing = sorted([a for a in unique_alleles if a not in mhc_seq_dict])
            if missing:
                raise KeyError(f"mhc_seq_dict is missing {len(missing)} allele(s), e.g. {missing[:10]}")
            self.mhc_token_cache = {}
            for allele in unique_alleles:
                seq = str(mhc_seq_dict[allele])
                if len(seq) > self.mhc_max_len:
                    raise ValueError(f"MHC pseudosequence for {allele} has length {len(seq)} > {self.mhc_max_len}.")
                self.mhc_token_cache[allele] = tokenizer.encode(seq)

            self.n_flank_tokens = torch.full((self.n_flank_len,), self.pad_idx, dtype=torch.long)
            self.c_flank_tokens = torch.full((self.c_flank_len,), self.pad_idx, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def _allele_at(self, idx):
            if self.allele_codes is not None:
                return str(self.allele_categories[int(self.allele_codes[idx])])
            return str(self.alleles[idx])

        def __getitem__(self, idx):
            raw_seq = str(self.seqs[idx]).rstrip("$")
            if len(raw_seq) > self.peptide_core_max_len:
                raise ValueError(
                    f"Peptide at row {idx} has length {len(raw_seq)} > {self.peptide_core_max_len}. "
                    "Expected unpadded peptide length <= 14."
                )

            peptide = self.tokenizer.encode(raw_seq)
            right_pad = torch.full(
                (self.peptide_core_max_len - len(peptide),), self.pad_idx, dtype=torch.long
            )
            # Exact legacy layout: N flank pad(10) + peptide + peptide-right-pad-to-14 + C flank pad(10).
            peptide_tokens = torch.cat([self.n_flank_tokens, peptide, right_pad, self.c_flank_tokens])

            allele_field = self._allele_at(idx)
            alleles = allele_field.split("_")
            mhc_list = torch.stack([self.mhc_token_cache[a] for a in alleles])

            item = {
                "peptide": peptide_tokens,
                "mhc_list": mhc_list,
                "label": torch.tensor(float(self.labels[idx]), dtype=torch.float),
                "allele": allele_field,
                "original_idx": int(idx),
                "domain_id": torch.tensor(int(self.domain_ids[idx]), dtype=torch.long),
            }
            if self.continuous_values is not None:
                item[f"peptide_{self.continuous_target_col}"] = torch.tensor(
                    float(self.continuous_values[idx]), dtype=torch.float
                )
            return item

    @staticmethod
    def _minimal_collate_fn(batch):
        max_peptide_len, max_mhc_len = 34, 34
        max_num_alleles = max(len(x["mhc_list"]) for x in batch)
        pad_idx = 1

        peptides = torch.stack([
            torch.cat([x["peptide"], torch.full((max_peptide_len - len(x["peptide"]),), pad_idx, dtype=torch.long)])
            for x in batch
        ])
        peptide_masks = peptides != pad_idx

        mhc_list = torch.stack([
            torch.cat([
                torch.cat([mhc, torch.full((max_mhc_len - len(mhc),), pad_idx, dtype=torch.long)]).unsqueeze(0)
                for mhc in x["mhc_list"]
            ] + [torch.full((1, max_mhc_len), pad_idx, dtype=torch.long)] * (max_num_alleles - len(x["mhc_list"])), dim=0)
            for x in batch
        ])
        mhc_mask_list = mhc_list != pad_idx

        result = {
            "peptide": peptides,
            "peptide_mask": peptide_masks,
            "mhc_list": mhc_list,
            "mhc_mask_list": mhc_mask_list,
            "label": torch.tensor([float(x["label"]) for x in batch], dtype=torch.float),
            "original_idx": torch.tensor([int(x["original_idx"]) for x in batch], dtype=torch.long),
            "allele": [x["allele"] for x in batch],
            "domain_id": torch.tensor([int(x["domain_id"]) for x in batch], dtype=torch.int),
        }
        for feature_key in batch[0].keys():
            if feature_key.startswith("peptide_") and feature_key not in {"peptide", "peptide_mask"}:
                vals = [x[feature_key] for x in batch if feature_key in x]
                if len(vals) == len(batch):
                    result[feature_key] = torch.tensor([float(v) for v in vals], dtype=torch.float)
        return result

    class _TargetOverrideDataset(Dataset):
        def __init__(self, parent_ds, domain_id, continuous_target=False, continuous_target_col="ba_value"):
            self.parent_ds = parent_ds
            self.domain_id = domain_id
            self.continuous_target = continuous_target
            self.item_continuous_key = f"peptide_{continuous_target_col}"

        def __len__(self):
            return len(self.parent_ds)

        def __getitem__(self, idx):
            item = dict(self.parent_ds[idx])
            item["domain_id"] = torch.tensor(self.domain_id, dtype=torch.long)
            if self.continuous_target:
                if self.item_continuous_key not in item:
                    raise KeyError(f"Continuous target key '{self.item_continuous_key}' missing from dataset item.")
                target = item[self.item_continuous_key]
                item["label"] = target.float() if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float)
            return item

    # =====================================================================
    # Losses
    # =====================================================================

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
        x_col, x_row = x.unsqueeze(-1), x.unsqueeze(-2)
        return 1 + torch.sigmoid((x_row - x_col) / tau).sum(-1)

    class _SoftSpearmanLoss(nn.Module):
        def __init__(self, tau: float = 0.1):
            super().__init__()
            self.tau = tau

        def forward(self, pred: torch.Tensor, target: torch.Tensor):
            if pred.dim() == 1:
                pred, target = pred.unsqueeze(0), target.unsqueeze(0)
            r_pred = PepCL_Logging_Ext._soft_rank(pred, tau=self.tau)
            r_target = PepCL_Logging_Ext._soft_rank(target, tau=1e-4)
            r_pred = r_pred - r_pred.mean(-1, keepdim=True)
            r_target = r_target - r_target.mean(-1, keepdim=True)
            num = (r_pred * r_target).sum(-1)
            den = torch.norm(r_pred, dim=-1) * torch.norm(r_target, dim=-1) + 1e-8
            return (1 - num / den).mean()

    # =====================================================================
    # Samplers: behavior preserved from PepCL_Logging
    # =====================================================================

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
            for (lbl, allele), group in subset.groupby(["label", "allele"], observed=True):
                lbl_int = int(lbl)
                if lbl_int not in self.pools:
                    continue
                idx = group.index.to_numpy()
                self.rng.shuffle(idx)
                self.pools[lbl_int][allele] = idx
            positive_alleles = sorted(list(self.pools[1].keys()), key=str)
            if self.q_pos_by_allele is not None:
                self.q_pos_by_allele = {al: int(q) for al, q in self.q_pos_by_allele.items() if int(q) > 0}
                self.alleles = [al for al in positive_alleles if self.q_pos_by_allele.get(al, 0) > 0]
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
                    requested_pos = int(self.q_pos_by_allele.get(al, 0)) if self.q_pos_by_allele is not None else int(self.n_pos)
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
                if len(indices) == 0:
                    return
                self.rng.shuffle(indices)
                yield from indices

    class _InfiniteBalancedSampler(Sampler):
        def __init__(self, df, domain_id, batch_size, seed):
            self.rng = np.random.default_rng(seed)
            self.batch_size = int(batch_size)
            subset = df[df["domain_id"] == domain_id]
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                return
            label_int = subset["label"].astype(int)
            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()
            self.rng.shuffle(self.pos_pool)
            self.rng.shuffle(self.neg_pool)

        def __iter__(self):
            if len(self.pos_pool) == 0 or len(self.neg_pool) == 0:
                return
            n_pos = self.batch_size // 2
            n_neg = self.batch_size - n_pos
            while True:
                sel_pos = self.rng.choice(self.pos_pool, size=n_pos, replace=len(self.pos_pool) < n_pos)
                sel_neg = self.rng.choice(self.neg_pool, size=n_neg, replace=len(self.neg_pool) < n_neg)
                indices = np.concatenate([sel_pos, sel_neg])
                self.rng.shuffle(indices)
                yield from indices

    class _InfinitePooledFTSampler(Sampler):
        def __init__(self, df, domain_id, pooled_num_pos, neg_ratio, batch_size, seed):
            self.rng = np.random.default_rng(seed)
            self.domain_id = domain_id
            self.neg_ratio = float(neg_ratio)
            self.batch_size = int(batch_size)
            if self.neg_ratio < 0:
                raise ValueError("neg_ratio must be >= 0.")
            subset = df[df["domain_id"] == domain_id]
            if subset.empty:
                self.pos_pool = np.array([], dtype=int)
                self.neg_pool = np.array([], dtype=int)
                self.pooled_num_pos = 0
                return
            label_int = subset["label"].astype(int)
            self.pos_pool = subset.index[label_int == 1].to_numpy()
            self.neg_pool = subset.index[label_int == 0].to_numpy()
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
                if len(indices) == 0:
                    return
                self.rng.shuffle(indices)
                yield from indices

    # =====================================================================
    # Logging helpers
    # =====================================================================

    def _init_nvml(self):
        """Initialize NVML for the exact CUDA device used by this PepCL instance.

        Prefer the CUDA device UUID because NVML indices are not guaranteed to match
        CUDA indices (e.g. under CUDA_VISIBLE_DEVICES). Fall back to the resolved CUDA
        device index only when UUID lookup is unavailable.
        """
        try:
            import pynvml
            pynvml.nvmlInit()

            device_index = self._device_index_from_any(self.device)

            # Resolve the physical device through CUDA first, then ask NVML for the same
            # UUID. This keeps telemetry attached to cuda:N actually used by the model.
            try:
                props = torch.cuda.get_device_properties(device_index)
                cuda_uuid = getattr(props, "uuid", None)
                if cuda_uuid is not None:
                    cuda_uuid = str(cuda_uuid)
                    handle = pynvml.nvmlDeviceGetHandleByUUID(cuda_uuid)
                    return (pynvml, handle)
            except Exception:
                pass

            # Fallback for environments/PyTorch versions that do not expose CUDA UUIDs.
            return (pynvml, pynvml.nvmlDeviceGetHandleByIndex(device_index))
        except Exception:
            return None

    @staticmethod
    def _device_index_from_any(device):
        try:
            d = torch.device(device)
            if d.type == "cuda":
                return torch.cuda.current_device() if d.index is None else int(d.index)
        except Exception:
            pass
        return torch.cuda.current_device() if torch.cuda.is_available() else 0

    @staticmethod
    def _gb(x):
        return None if x is None else float(x) / (1024 ** 3)

    @staticmethod
    def _df_bytes(df):
        try:
            return int(df.memory_usage(index=True, deep=True).sum())
        except Exception:
            return None

    def _memory_full_info_safe(self, proc):
        try:
            info = proc.memory_full_info()
            return {
                "rss": getattr(info, "rss", None),
                "uss": getattr(info, "uss", None),
                "pss": getattr(info, "pss", None),
                "vms": getattr(info, "vms", None),
            }
        except Exception:
            try:
                info = proc.memory_info()
                return {"rss": getattr(info, "rss", None), "uss": None, "pss": None, "vms": getattr(info, "vms", None)}
            except Exception:
                return {"rss": None, "uss": None, "pss": None, "vms": None}

    def _environment_stats(self):
        env = {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python_version": sys.version.replace("\n", " "),
            "torch_version": torch.__version__,
            "cuda_runtime_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cpu_model": platform.processor() or platform.machine(),
            "cpu_logical_count": int(psutil.cpu_count(logical=True)) if psutil is not None else (os.cpu_count() or None),
            "cpu_physical_count": int(psutil.cpu_count(logical=False)) if psutil is not None and psutil.cpu_count(logical=False) is not None else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        }
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                env["system_ram_total_gb"] = self._gb(vm.total)
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                idx = self._device_index_from_any(self.device if hasattr(self, "device") else "cuda")
                props = torch.cuda.get_device_properties(idx)
                env.update({
                    "gpu_index": int(idx),
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "gpu_total_memory_gb": self._gb(props.total_memory),
                    "gpu_compute_capability": f"{props.major}.{props.minor}",
                })
            except Exception:
                pass
        return env

    def _snapshot(self, tag, step=None, detailed_memory=False):
        if not self.enable_logging:
            return
        snap = {"tag": tag, "step": step, "elapsed_since_object_init_s": time.perf_counter() - self._object_t0}
        if self._process is not None:
            try:
                # Cheap RSS/system stats are recorded at every snapshot. PSS/USS are intentionally
                # collected only for major phase snapshots to avoid /proc/smaps overhead during training.
                mi = self._process.memory_info()
                snap["process_rss_gb"] = self._gb(mi.rss)
                snap["process_vms_gb"] = self._gb(mi.vms)
                children = self._process.children(recursive=True)
                child_rss = 0
                for c in children:
                    try:
                        child_rss += c.memory_info().rss
                    except Exception:
                        pass
                snap["child_processes"] = len(children)
                snap["children_rss_gb"] = self._gb(child_rss)
                snap["process_tree_rss_gb"] = self._gb(mi.rss + child_rss)
                snap["process_cpu_percent"] = self._process.cpu_percent(interval=None)

                if detailed_memory:
                    pinfo = self._memory_full_info_safe(self._process)
                    snap["process_uss_gb"] = self._gb(pinfo["uss"])
                    snap["process_pss_gb"] = self._gb(pinfo["pss"])
                    child_uss = child_pss = 0
                    child_uss_valid = child_pss_valid = False
                    for c in children:
                        ci = self._memory_full_info_safe(c)
                        if ci["uss"] is not None:
                            child_uss += ci["uss"]
                            child_uss_valid = True
                        if ci["pss"] is not None:
                            child_pss += ci["pss"]
                            child_pss_valid = True
                    snap["children_uss_gb"] = self._gb(child_uss) if child_uss_valid else None
                    snap["children_pss_gb"] = self._gb(child_pss) if child_pss_valid else None
                    snap["process_tree_uss_gb"] = (
                        (snap.get("process_uss_gb") or 0.0) + (snap.get("children_uss_gb") or 0.0)
                        if snap.get("process_uss_gb") is not None or snap.get("children_uss_gb") is not None else None
                    )
                    snap["process_tree_pss_gb"] = (
                        (snap.get("process_pss_gb") or 0.0) + (snap.get("children_pss_gb") or 0.0)
                        if snap.get("process_pss_gb") is not None or snap.get("children_pss_gb") is not None else None
                    )

                vm = psutil.virtual_memory()
                snap["system_ram_used_gb"] = self._gb(vm.used)
                snap["system_ram_available_gb"] = self._gb(vm.available)
                snap["system_ram_percent"] = float(vm.percent)
            except Exception:
                pass

        # Baseline-relative values are much more interpretable than absolute server utilization.
        baseline = self.logs.get("memory_baseline", {}) if hasattr(self, "logs") else {}
        for field in (
            "process_rss_gb", "process_uss_gb", "process_pss_gb",
            "process_tree_rss_gb", "process_tree_uss_gb", "process_tree_pss_gb",
            "system_ram_used_gb"
        ):
            if field in snap and snap.get(field) is not None and baseline.get(field) is not None:
                snap[f"{field}_delta_from_baseline"] = snap[field] - baseline[field]

        if torch.cuda.is_available():
            try:
                dev = self.device if hasattr(self, "device") else torch.device("cuda", torch.cuda.current_device())
                snap["gpu_memory_allocated_gb"] = self._gb(torch.cuda.memory_allocated(dev))
                snap["gpu_memory_reserved_gb"] = self._gb(torch.cuda.memory_reserved(dev))
                snap["gpu_peak_allocated_gb"] = self._gb(torch.cuda.max_memory_allocated(dev))
                snap["gpu_peak_reserved_gb"] = self._gb(torch.cuda.max_memory_reserved(dev))
                free_b, total_b = torch.cuda.mem_get_info(dev)
                snap["gpu_free_gb"] = self._gb(free_b)
                snap["gpu_total_gb"] = self._gb(total_b)
            except Exception:
                pass
        if self._nvml is not None:
            try:
                pynvml, handle = self._nvml
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                snap["gpu_util_percent"] = float(util.gpu)
                snap["gpu_memory_util_percent"] = float(util.memory)
                try:
                    snap["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass
            except Exception:
                pass
        self.logs["memory_snapshots"].append(snap)

    def _telemetry(self, step, step_wall_s, completed_steps, train_loop_t0):
        if not self.enable_logging:
            return
        if step != 1 and step % self.telemetry_every_n_steps != 0 and step != self.steps:
            return
        elapsed = time.perf_counter() - train_loop_t0
        row = {
            "step": int(step), "step_wall_seconds": float(step_wall_s),
            "cumulative_train_seconds": float(elapsed),
            "cumulative_steps_per_second": float(completed_steps / elapsed) if elapsed > 0 else np.nan,
        }
        self._snapshot(f"train_step_{step}", step=step, detailed_memory=False)
        if self.logs["memory_snapshots"]:
            row.update({k: v for k, v in self.logs["memory_snapshots"][-1].items() if k not in {"tag", "step"}})
        self.logs["step_telemetry"].append(row)

    def _model_stats(self, model, prefix):
        params = list(model.parameters())
        buffers = list(model.buffers())
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in params)
        buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
        out = {
            f"{prefix}_parameter_count": int(total_params),
            f"{prefix}_trainable_parameter_count": int(trainable_params),
            f"{prefix}_parameter_bytes": int(param_bytes),
            f"{prefix}_buffer_bytes": int(buffer_bytes),
        }
        if self.measure_serialized_model_size:
            t = time.perf_counter()
            bio = io.BytesIO()
            torch.save(model.state_dict(), bio)
            out[f"{prefix}_serialized_state_dict_bytes"] = int(bio.getbuffer().nbytes)
            self.logs["timing"][f"{prefix}_serialized_size_measure_seconds"] = time.perf_counter() - t
        return out

    def get_logs(self, as_dataframes=False):
        out = copy.deepcopy(self.logs)
        out["compute"] = dict(out["compute"])
        if as_dataframes:
            out["memory_snapshots"] = pd.DataFrame(out["memory_snapshots"])
            out["step_telemetry"] = pd.DataFrame(out["step_telemetry"])
        return out

    # =====================================================================
    # Data preparation
    # =====================================================================

    def _prepare_compact_dataframe(self, train_df):
        required = ["seq", "allele", "label", "domain_id"]
        if self.ft_is_continuous:
            required.append(self.continuous_target_col)
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise ValueError(f"train_df is missing required minimal columns: {missing}")

        # Use a shallow column selection so the large sequence object array is not duplicated.
        # Converted compact columns (allele/label/domain/continuous target) are replaced independently.
        if self.compact_dataframe:
            df = train_df.loc[:, required].copy(deep=False)
        else:
            df = train_df.copy(deep=False)
        if not (isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1 and df.index.stop == len(df)):
            df = df.reset_index(drop=True)

        if self.compact_dtypes:
            # No row values are changed; only storage dtypes are compacted.
            if not isinstance(df["allele"].dtype, pd.CategoricalDtype):
                df["allele"] = df["allele"].astype("category")
            # Binary labels remain exact under uint8. If non-binary labels are unexpectedly present, keep float32.
            unique_labels = pd.unique(df["label"].dropna())
            if len(unique_labels) <= 2 and set(np.asarray(unique_labels, dtype=float).tolist()).issubset({0.0, 1.0}):
                df["label"] = df["label"].astype(np.uint8)
            else:
                df["label"] = df["label"].astype(np.float32)
            min_domain, max_domain = int(df["domain_id"].min()), int(df["domain_id"].max())
            if -128 <= min_domain and max_domain <= 127:
                df["domain_id"] = df["domain_id"].astype(np.int8)
            else:
                df["domain_id"] = df["domain_id"].astype(np.int16)
            if self.ft_is_continuous:
                df[self.continuous_target_col] = df[self.continuous_target_col].astype(np.float32)
        return df

    # =====================================================================
    # Criterion / optimizer / sampling calibration
    # =====================================================================

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
            [{"params": e, "lr": self.encoder_lr}, {"params": d, "lr": self.decoder_lr}],
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def _integer_quota_allocation(weights, caps, total, ensure_all_if_possible=True):
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
        w[valid] = weights[valid] / weights[valid].sum() if weights[valid].sum() > 0 else 1.0 / valid.sum()
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
            w_active[can_receive] = (
                w[can_receive] / w[can_receive].sum() if w[can_receive].sum() > 0 else 1.0 / can_receive.sum()
            )
            raw = remaining_total * w_active
            add = np.minimum(np.floor(raw).astype(int), capacity)
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
        sub = self.df[self.df["domain_id"] == domain_id]
        if sub.empty:
            return {}, pd.DataFrame()
        counts = sub.groupby(["allele", "label"], observed=True).size().unstack(fill_value=0)
        n_pos = counts[1] if 1 in counts.columns else pd.Series(0, index=counts.index)
        n_neg = counts[0] if 0 in counts.columns else pd.Series(0, index=counts.index)
        summary = pd.DataFrame({
            "allele": counts.index.astype(str),
            "n_pos": n_pos.astype(int).to_numpy(),
            "n_neg": n_neg.astype(int).to_numpy(),
        })
        summary = summary[summary["n_pos"] > 0].reset_index(drop=True)
        if summary.empty:
            return {}, summary
        A = len(summary)
        P = summary["n_pos"].astype(float).to_numpy()
        caps = summary["n_pos"].astype(int).to_numpy()
        target_pos = max(1, int(batch_size / (1.0 + float(neg_ratio))))
        uniform = np.ones(A, dtype=float) / A
        prevalence = P / P.sum()
        denom = np.square(P).sum()
        A_eff = float(A) if denom <= 0 else float((P.sum() ** 2) / denom)
        balance_score = float(np.clip(A_eff / A, 0.0, 1.0))
        weights = balance_score * uniform + (1.0 - balance_score) * prevalence
        weights /= weights.sum()
        q_pos = self._integer_quota_allocation(weights, caps, target_pos, ensure_all_if_possible=False)
        q_pos_by_allele = {al: int(q) for al, q in zip(summary["allele"].tolist(), q_pos) if int(q) > 0}
        summary["allele_weight"] = weights
        summary["q_pos"] = q_pos.astype(int)
        summary["expected_q_neg_if_available"] = np.floor(summary["q_pos"].to_numpy() * float(neg_ratio)).astype(int)
        summary["reuse_pressure_pos"] = summary["q_pos"] / summary["n_pos"].clip(lower=1)
        summary["has_negatives"] = summary["n_neg"] > 0
        summary.attrs.update({"target_pos": target_pos, "n_alleles": A, "effective_n_alleles": A_eff, "balance_score": balance_score})
        return q_pos_by_allele, summary

    def _compute_pooled_ft_summary(self):
        sub = self.df[self.df["domain_id"] == 1]
        if sub.empty:
            return pd.DataFrame()
        label_int = sub["label"].astype(int)
        n_pos_total = int((label_int == 1).sum())
        n_neg_total = int((label_int == 0).sum())
        n_alleles_pos = int(sub.loc[label_int == 1, "allele"].nunique())
        n_alleles_all = int(sub["allele"].nunique())
        pooled_pos = max(1, int(self.ft_cfg["bs"] / (1.0 + float(self.ft_cfg["ratio"])))) if self.pooled_ft_num_pos is None else int(self.pooled_ft_num_pos)
        pooled_neg = int(pooled_pos * float(self.ft_cfg["ratio"]))
        summary = pd.DataFrame({
            "mode": ["pooled"], "n_pos_total": [n_pos_total], "n_neg_total": [n_neg_total],
            "n_alleles_pos": [n_alleles_pos], "n_alleles_all": [n_alleles_all],
            "pooled_ft_num_pos": [pooled_pos], "expected_pooled_num_neg": [pooled_neg],
            "pos_reuse_pressure": [pooled_pos / max(n_pos_total, 1)],
        })
        summary.attrs["pooled_ft_num_pos"] = pooled_pos
        summary.attrs["expected_pooled_num_neg"] = pooled_neg
        return summary

    def _auto_calibrate(self):
        # This parameter is functional in PepCL_Logging. Only active streams are calibrated here.
        if self.need_ms_stream:
            ms_q, ms_summary = self._compute_auto_domain_sampling(0, self.ms_cfg["bs"], self.ms_cfg["ratio"])
            self.ms_cfg["q_pos_by_allele"] = ms_q
            self.ms_cfg["sampling_summary"] = ms_summary
        if self.need_ft_stream:
            if self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
                self.ft_cfg["q_pos_by_allele"] = None
                self.ft_cfg["sampling_summary"] = self._compute_pooled_ft_summary()
            else:
                ft_q, ft_summary = self._compute_auto_domain_sampling(1, self.ft_cfg["bs"], self.ft_cfg["ratio"])
                self.ft_cfg["q_pos_by_allele"] = ft_q
                self.ft_cfg["sampling_summary"] = ft_summary
        self.logs["sampling"]["ms_summary"] = self.ms_cfg["sampling_summary"]
        self.logs["sampling"]["ft_summary"] = self.ft_cfg["sampling_summary"]

        def _summary_line(name, summary, cfg):
            if summary is None or summary.empty:
                return f"  {name}: inactive or no eligible positive alleles"
            if "mode" in summary.columns and summary["mode"].iloc[0] == "pooled":
                row = summary.iloc[0]
                return (
                    f"  {name}: pooled continuous FT, alleles_pos={int(row['n_alleles_pos'])}, "
                    f"alleles_all={int(row['n_alleles_all'])}, pos_pool={int(row['n_pos_total'])}, "
                    f"neg_pool={int(row['n_neg_total'])}, pooled_pos/cycle={int(row['pooled_ft_num_pos'])}, "
                    f"expected_neg/cycle={int(row['expected_pooled_num_neg'])}, neg_ratio={cfg['ratio']}"
                )
            target_pos = summary.attrs.get("target_pos")
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
                f"  {name}: alleles={A}, A_eff={A_eff:.2f}, balance={balance_score:.3f}, target_pos/cycle={target_pos}, "
                f"q_pos median={q_med:.1f} [{q_min}, {q_max}], reuse median={reuse_med:.4f}, "
                f"reuse max={reuse_max:.4f}, neg_ratio={cfg['ratio']}"
            )
        print("[PepCL_Logging_Ext] Auto-calibrated sampling:")
        if self.need_ms_stream:
            print(_summary_line("MS", self.ms_cfg["sampling_summary"], self.ms_cfg))
        if self.need_ft_stream:
            print(_summary_line("FT", self.ft_cfg["sampling_summary"], self.ft_cfg))
        print("")

    # =====================================================================
    # Loader construction
    # =====================================================================

    def _build_loader(self, domain_id, cfg, seed):
        continuous_target = domain_id == 1 and self.ft_is_continuous
        ds = self._TargetOverrideDataset(
            self.base_dataset, domain_id=domain_id, continuous_target=continuous_target,
            continuous_target_col=self.continuous_target_col,
        )
        if not self.allele_aware_sampling:
            sampler = self._InfiniteBalancedSampler(self.df, domain_id, cfg["bs"], seed)
        elif domain_id == 1 and self.ft_is_continuous and cfg.get("sampling_mode", "allele_aware") == "pooled":
            sampler = self._InfinitePooledFTSampler(
                self.df, domain_id, cfg.get("pooled_ft_num_pos"), cfg["ratio"], cfg["bs"], seed
            )
        else:
            sampler = self._InfiniteAlleleSampler(
                self.df, domain_id, cfg["n_pos"], cfg["ratio"], seed,
                q_pos_by_allele=cfg.get("q_pos_by_allele"),
            )
        return DataLoader(
            ds, batch_size=cfg["bs"], sampler=sampler, num_workers=self.num_workers,
            collate_fn=self.collate_fn, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @staticmethod
    def _loader_stats(loader):
        return {
            "batch_size": getattr(loader, "batch_size", None),
            "num_workers": getattr(loader, "num_workers", None),
            "pin_memory": getattr(loader, "pin_memory", None),
            "persistent_workers": getattr(loader, "persistent_workers", None),
            "prefetch_factor": getattr(loader, "prefetch_factor", None),
            "drop_last": getattr(loader, "drop_last", None),
            "sampler_class": type(getattr(loader, "sampler", None)).__name__,
            "collate_fn": getattr(getattr(loader, "collate_fn", None), "__name__", type(getattr(loader, "collate_fn", None)).__name__),
        }

    def _move_batch(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # =====================================================================
    # Training
    # =====================================================================

    def train(self):
        train_call_t0 = time.perf_counter()
        self._snapshot("train_enter", detailed_memory=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Sampling setup.
        t = time.perf_counter()
        if not self.allele_aware_sampling:
            print(
                "[PepCL_Logging_Ext] Naive balanced sampling: allele-aware sampling disabled; "
                "each active stream samples 1:1 positives/negatives globally. num_*_pos and *_neg_ratio are ignored.\n"
            )
        elif self.auto_calibrate:
            self._auto_calibrate()
        else:
            if self.need_ft_stream and self.ft_is_continuous and self.continuous_ft_sampling_mode == "pooled":
                self.ft_cfg["sampling_summary"] = self._compute_pooled_ft_summary()
            print(
                f"[PepCL_Logging_Ext] Manual mode: MS n_pos={self.ms_cfg['n_pos']}, MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, FT neg_ratio={self.ft_cfg['ratio']} | "
                f"FT continuous={self.ft_is_continuous}, FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )
        self.logs["timing"]["sampling_setup_seconds"] = time.perf_counter() - t
        self._snapshot("after_sampling_setup", detailed_memory=True)

        # Build only streams actually used. This is what allows FT-only runs to omit domain 0 completely.
        ms_iter = ft_iter = None
        t = time.perf_counter()
        if self.need_ms_stream:
            ms_loader = self._build_loader(0, self.ms_cfg, self.seed)
            self.logs["timing"]["ms_loader_construct_seconds"] = time.perf_counter() - t
            self.logs["data_loader"]["ms"] = self._loader_stats(ms_loader)
            t_iter = time.perf_counter()
            ms_iter = iter(ms_loader)
            self.logs["timing"]["ms_loader_iterator_start_seconds"] = time.perf_counter() - t_iter
        else:
            self.logs["timing"]["ms_loader_construct_seconds"] = 0.0
            self.logs["timing"]["ms_loader_iterator_start_seconds"] = 0.0

        t = time.perf_counter()
        if self.need_ft_stream:
            ft_loader = self._build_loader(1, self.ft_cfg, self.seed + 1)
            self.logs["timing"]["ft_loader_construct_seconds"] = time.perf_counter() - t
            self.logs["data_loader"]["ft"] = self._loader_stats(ft_loader)
            t_iter = time.perf_counter()
            ft_iter = iter(ft_loader)
            self.logs["timing"]["ft_loader_iterator_start_seconds"] = time.perf_counter() - t_iter
        else:
            self.logs["timing"]["ft_loader_construct_seconds"] = 0.0
            self.logs["timing"]["ft_loader_iterator_start_seconds"] = 0.0
        self._snapshot("after_loader_iterator_start", detailed_memory=True)

        # Optimizer / scheduler / scaler setup.
        t = time.perf_counter()
        optimizer = self._get_optimizer()
        warmup_steps = int(self.steps * self.warmup_ratio)
        if self.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.steps)
        elif self.lr_scheduler_type == "warmup_constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
        else:
            scheduler = get_constant_schedule(optimizer)
        scaler = GradScaler()
        self.logs["timing"]["optimizer_scheduler_scaler_setup_seconds"] = time.perf_counter() - t
        self._snapshot("after_optimizer_setup_before_state_allocation", detailed_memory=True)

        metrics = collections.defaultdict(list)
        model_checkpoints = {} if self.save_every_n_steps is not None and self.output_dir is None else None
        pbar = tqdm(range(self.steps), desc="PepCL_Logging_Ext")

        compute = self.logs["compute"]
        train_loop_t0 = time.perf_counter()
        first_optimizer_snapshot_done = False
        step_wall_times = []

        for step in pbar:
            step_t0 = time.perf_counter()
            self.model.train()
            optimizer.zero_grad()

            # Legacy display/return values retained for backward compatibility.
            l_ms_v = l_ft_v = l_dr_v = 0.0

            # Granular raw/weighted loss tensors for this optimizer step.
            loss_ms_cls_raw = loss_ms_cls_weighted = None
            loss_ft_raw = loss_ft_weighted = None
            loss_dr_raw = loss_dr_weighted = None
            loss_ms = loss_ft = total_l = None

            # Domain 0 / replay and/or drift regularization.
            if self.need_ms_stream:
                try:
                    t_fetch = time.perf_counter()
                    batch_ms = next(ms_iter)
                    compute["ms_batch_fetch_seconds"] += time.perf_counter() - t_fetch
                    t_move = time.perf_counter()
                    batch_ms = self._move_batch(batch_ms)
                    compute["ms_host_to_device_seconds"] += time.perf_counter() - t_move
                    lbl_ms = batch_ms.pop("label")
                    ms_n = int(lbl_ms.numel())
                    compute["ms_batches"] += 1
                    compute["ms_samples"] += ms_n

                    with autocast(self.device.type):
                        logits_ms = self.model(batch_ms).view(-1)
                        compute["student_forward_passes_ms"] += 1
                        compute["student_forward_samples_ms"] += ms_n

                        if self.compute_ms_classification:
                            loss_ms_cls_raw = self.criterion_ms(logits_ms, lbl_ms)
                            loss_ms_cls_weighted = loss_ms_cls_raw * self.ms_lambda

                        if self.dr_lambda > 0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits = self.teacher(batch_ms).view(-1)
                            compute["teacher_forward_passes"] += 1
                            compute["teacher_forward_samples"] += ms_n
                            loss_dr_raw = self._get_dr_loss(logits_ms, t_logits)
                            loss_dr_weighted = loss_dr_raw * self.dr_lambda

                        if loss_ms_cls_weighted is not None and loss_dr_weighted is not None:
                            loss_ms = loss_ms_cls_weighted + loss_dr_weighted
                        elif loss_ms_cls_weighted is not None:
                            loss_ms = loss_ms_cls_weighted
                        else:
                            loss_ms = loss_dr_weighted

                    if self.seq_backprop and loss_ms is not None:
                        scaler.scale(loss_ms).backward()
                        compute["backward_passes"] += 1

                except (StopIteration, AttributeError):
                    pass

            # Domain 1 / update stream. The criterion may be classification or
            # continuous (Spearman/MSE); raw and weighted values are logged identically.
            if self.need_ft_stream:
                try:
                    t_fetch = time.perf_counter()
                    batch_ft = next(ft_iter)
                    compute["ft_batch_fetch_seconds"] += time.perf_counter() - t_fetch
                    t_move = time.perf_counter()
                    batch_ft = self._move_batch(batch_ft)
                    compute["ft_host_to_device_seconds"] += time.perf_counter() - t_move
                    lbl_ft = batch_ft.pop("label")
                    ft_n = int(lbl_ft.numel())
                    compute["ft_batches"] += 1
                    compute["ft_samples"] += ft_n

                    with autocast(self.device.type):
                        logits_ft = self.model(batch_ft).view(-1)
                        compute["student_forward_passes_ft"] += 1
                        compute["student_forward_samples_ft"] += ft_n
                        loss_ft_raw = self.criterion_ft(logits_ft, lbl_ft)
                        loss_ft_weighted = loss_ft_raw * self.ft_lambda
                        loss_ft = loss_ft_weighted

                    if self.seq_backprop and loss_ft is not None:
                        scaler.scale(loss_ft).backward()
                        compute["backward_passes"] += 1

                except (StopIteration, AttributeError):
                    pass

            # The total optimization objective for this step is always the sum of
            # the weighted active components, regardless of whether their backward
            # passes are executed sequentially or jointly.
            if loss_ms is not None:
                total_l = loss_ms
            if loss_ft is not None:
                total_l = loss_ft if total_l is None else total_l + loss_ft

            if not self.seq_backprop and total_l is not None:
                scaler.scale(total_l).backward()
                compute["backward_passes"] += 1

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            compute["optimizer_steps"] += 1
            compute["scheduler_steps"] += 1

            if not first_optimizer_snapshot_done:
                self._snapshot("after_first_optimizer_step_state_allocated", step=1, detailed_memory=True)
                first_optimizer_snapshot_done = True

            # Convert only scalar losses to Python values for logging. Missing/inactive
            # components are stored as NaN in the granular trajectories so they are
            # distinguishable from a true zero-valued loss. Legacy keys retain 0.0
            # for inactive components to preserve prior behavior.
            def _loss_value(x):
                return float(x.detach().item()) if x is not None else np.nan

            ms_cls_raw_v = _loss_value(loss_ms_cls_raw)
            ms_cls_weighted_v = _loss_value(loss_ms_cls_weighted)
            ft_raw_v = _loss_value(loss_ft_raw)
            ft_weighted_v = _loss_value(loss_ft_weighted)
            dr_raw_v = _loss_value(loss_dr_raw)
            dr_weighted_v = _loss_value(loss_dr_weighted)
            total_loss_v = _loss_value(total_l)

            # Preserve historical ms_loss / ft_loss / dr_loss semantics.
            if self.compute_ms_classification and loss_ms is not None:
                l_ms_v = _loss_value(loss_ms)
            if loss_ft_weighted is not None:
                l_ft_v = ft_weighted_v
            if loss_dr_weighted is not None:
                l_dr_v = dr_weighted_v

            pbar.set_postfix({"MS": f"{l_ms_v:.6f}", "FT": f"{l_ft_v:.6f}", "DR": f"{l_dr_v:.6f}"})

            # Legacy trajectories.
            metrics["ms_loss"].append(l_ms_v)
            metrics["ft_loss"].append(l_ft_v)
            metrics["dr_loss"].append(l_dr_v)

            # Granular trajectories requested for ablation/diagnostic analysis.
            metrics["ms_classification_loss_raw"].append(ms_cls_raw_v)
            metrics["ms_classification_loss_weighted"].append(ms_cls_weighted_v)
            metrics["ft_loss_raw"].append(ft_raw_v)
            metrics["ft_loss_weighted"].append(ft_weighted_v)
            metrics["drift_loss_raw"].append(dr_raw_v)
            metrics["drift_loss_weighted"].append(dr_weighted_v)
            metrics["total_loss"].append(total_loss_v)

            # Explicit aliases for continuous FT runs. These are NaN for
            # classification FT, and exactly mirror FT raw/weighted losses when
            # the FT criterion is Spearman or MSE.
            metrics["continuous_loss_raw"].append(ft_raw_v if self.ft_is_continuous else np.nan)
            metrics["continuous_loss_weighted"].append(ft_weighted_v if self.ft_is_continuous else np.nan)

            completed_step = step + 1
            if self.save_every_n_steps is None:
                should_save = False
            elif isinstance(self.save_every_n_steps, int):
                should_save = completed_step % self.save_every_n_steps == 0
            else:
                should_save = completed_step in self.save_every_n_steps

            if should_save:
                t_save = time.perf_counter()
                if self.output_dir is not None:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"pepcl_step_{completed_step}.pt"))
                else:
                    model_checkpoints[completed_step] = copy.deepcopy(self.model).cpu().eval()
                compute["checkpoint_count"] += 1
                compute["checkpoint_seconds"] += time.perf_counter() - t_save
                self._snapshot(f"after_checkpoint_{completed_step}", step=completed_step)

            step_wall_s = time.perf_counter() - step_t0
            step_wall_times.append(step_wall_s)
            compute["sum_step_wall_seconds"] += step_wall_s
            self._telemetry(completed_step, step_wall_s, completed_step, train_loop_t0)

        # High-level synchronized timing at the end only; no per-step cuda synchronize is introduced.
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        train_loop_seconds = time.perf_counter() - train_loop_t0
        self.logs["timing"]["training_loop_seconds"] = train_loop_seconds
        self.logs["timing"]["train_call_total_seconds"] = time.perf_counter() - train_call_t0
        self.logs["timing"]["end_to_end_seconds"] = time.perf_counter() - self._object_t0
        self.logs["timing"]["average_steps_per_second"] = self.steps / train_loop_seconds if train_loop_seconds > 0 else np.nan
        self.logs["timing"]["average_seconds_per_step"] = train_loop_seconds / self.steps if self.steps > 0 else np.nan
        self.logs["timing"]["first_step_seconds"] = float(step_wall_times[0]) if step_wall_times else np.nan
        if len(step_wall_times) > 10:
            steady_s = float(sum(step_wall_times[10:]))
            steady_n = len(step_wall_times) - 10
            self.logs["timing"]["steady_state_start_step"] = 11
            self.logs["timing"]["steady_state_steps_per_second_after_step_10"] = steady_n / steady_s if steady_s > 0 else np.nan
            self.logs["timing"]["steady_state_seconds_per_step_after_step_10"] = steady_s / steady_n if steady_n > 0 else np.nan
        else:
            self.logs["timing"]["steady_state_start_step"] = None
            self.logs["timing"]["steady_state_steps_per_second_after_step_10"] = np.nan
            self.logs["timing"]["steady_state_seconds_per_step_after_step_10"] = np.nan
        compute["student_forward_passes_total"] = compute["student_forward_passes_ms"] + compute["student_forward_passes_ft"]
        compute["student_forward_samples_total"] = compute["student_forward_samples_ms"] + compute["student_forward_samples_ft"]
        compute["total_forward_passes_including_teacher"] = compute["student_forward_passes_total"] + compute["teacher_forward_passes"]
        compute["total_forward_samples_including_teacher"] = compute["student_forward_samples_total"] + compute["teacher_forward_samples"]
        compute["student_samples_per_second"] = compute["student_forward_samples_total"] / train_loop_seconds if train_loop_seconds > 0 else np.nan
        compute["ms_samples_per_second"] = compute["student_forward_samples_ms"] / train_loop_seconds if train_loop_seconds > 0 else np.nan
        compute["ft_samples_per_second"] = compute["student_forward_samples_ft"] / train_loop_seconds if train_loop_seconds > 0 else np.nan
        compute["teacher_samples_per_second"] = compute["teacher_forward_samples"] / train_loop_seconds if train_loop_seconds > 0 else np.nan
        compute["seconds_per_1000_student_samples"] = (1000.0 * train_loop_seconds / compute["student_forward_samples_total"]) if compute["student_forward_samples_total"] > 0 else np.nan

        self._snapshot("train_exit", step=self.steps, detailed_memory=True)
        if torch.cuda.is_available():
            self.logs["model"]["train_peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
            self.logs["model"]["train_peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self.device))
            try:
                self.logs["cuda_allocator_stats_end"] = {k: int(v) for k, v in torch.cuda.memory_stats(self.device).items()}
            except Exception:
                self.logs["cuda_allocator_stats_end"] = {}

        # Explicit peak deltas relative to the pre-constructor baseline. For system RAM this is
        # the most operationally meaningful host-memory quantity on a shared server.
        snaps = self.logs.get("memory_snapshots", [])
        baseline = self.logs.get("memory_baseline", {})
        if snaps:
            def _peak_delta(field):
                vals = [x.get(field) for x in snaps if x.get(field) is not None]
                b = baseline.get(field)
                return (max(vals) - b) if vals and b is not None else None
            self.logs["model"]["peak_system_ram_delta_gb"] = _peak_delta("system_ram_used_gb")
            self.logs["model"]["peak_process_rss_delta_gb"] = _peak_delta("process_rss_gb")
            self.logs["model"]["peak_process_tree_rss_delta_gb"] = _peak_delta("process_tree_rss_gb")
            self.logs["model"]["peak_process_uss_delta_gb"] = _peak_delta("process_uss_gb")
            self.logs["model"]["peak_process_pss_delta_gb"] = _peak_delta("process_pss_gb")
            self.logs["model"]["peak_process_tree_uss_delta_gb"] = _peak_delta("process_tree_uss_gb")
            self.logs["model"]["peak_process_tree_pss_delta_gb"] = _peak_delta("process_tree_pss_gb")

        # FLOPs are intentionally not profiled during the benchmark run: torch.profiler FLOP accounting
        # is operator-limited and profiling perturbs timing. Exact forward/backward counts above are logged instead.
        self.logs["notes"].append(
            "FLOPs are not profiled inside the timed run. PyTorch profiler with_flops estimates only selected operators; "
            "run FLOP profiling separately if required so benchmark timing is not perturbed."
        )

        if model_checkpoints is not None:
            return metrics, model_checkpoints
        return metrics


class PepCL_Logging_Ext_Grad_Norm(PepCL_Logging_Ext):
    """
    PepCL_Logging_Ext + isolated component-gradient diagnostics.

    Training behavior is inherited from PepCL_Logging_Ext and preserved.
    Gradient diagnostics use torch.autograd.grad(), which does NOT accumulate
    into parameter .grad fields. The normal PepCL backward/optimizer path is
    therefore unchanged.

    Logged component gradients:
        replay/MS classification:
            ∇ L_MS_raw
            ∇ (ms_lambda * L_MS_raw)

        drift regularization:
            ∇ L_DR_raw
            ∇ (dr_lambda * L_DR_raw)

        FT/update:
            ∇ L_FT_raw
            ∇ (ft_lambda * L_FT_raw)

    Also logs pairwise gradient dot products, allowing post-hoc cosine
    similarities without retaining full gradient vectors.

    Gradient logging is computationally expensive because isolated component
    gradients require additional backward-like autograd traversals. It is
    therefore sampled by default rather than run on every optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,
        mhc_seq_dict: dict,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps=None,
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
        allele_aware_sampling: bool = True,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
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
        dr_loss_type: str = "mse",

        compact_dataframe: bool = True,
        compact_dtypes: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,

        enable_logging: bool = True,
        telemetry_every_n_steps: int = 10,
        enable_nvml_sampling: bool = True,
        measure_serialized_model_size: bool = False,

        # --------------------------------------------------------------
        # Gradient diagnostics.
        # --------------------------------------------------------------
        enable_gradient_logging: bool = True,
        gradient_every_n_steps: int = 10,
        gradient_steps=None,
        gradient_log_dot_products: bool = True,
    ):
        super().__init__(
            model=model,
            train_df=train_df,
            tokenizer=tokenizer,
            device=device,
            mhc_seq_dict=mhc_seq_dict,

            steps=steps,
            output_dir=output_dir,
            save_every_n_steps=save_every_n_steps,
            seed=seed,
            num_workers=num_workers,
            collate_fn=collate_fn,

            num_ms_pos=num_ms_pos,
            ms_neg_ratio=ms_neg_ratio,
            num_nonms_pos=num_nonms_pos,
            nonms_neg_ratio=nonms_neg_ratio,
            ms_batch_size=ms_batch_size,
            nonms_batch_size=nonms_batch_size,

            auto_calibrate=auto_calibrate,
            calibration_steps=calibration_steps,

            continuous_target_col=continuous_target_col,
            continuous_ft_sampling_mode=continuous_ft_sampling_mode,
            pooled_ft_num_pos=pooled_ft_num_pos,
            allele_aware_sampling=allele_aware_sampling,

            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            update_domain_embeddings=update_domain_embeddings,

            use_replay=use_replay,
            primary_task_id=primary_task_id,
            sequential_backprop=sequential_backprop,

            ms_loss_type=ms_loss_type,
            ms_loss_alpha=ms_loss_alpha,
            ms_lambda=ms_lambda,
            nonms_loss_type=nonms_loss_type,
            nonms_loss_alpha=nonms_loss_alpha,
            ft_lambda=ft_lambda,
            dr_lambda=dr_lambda,
            dr_loss_type=dr_loss_type,

            compact_dataframe=compact_dataframe,
            compact_dtypes=compact_dtypes,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,

            enable_logging=enable_logging,
            telemetry_every_n_steps=telemetry_every_n_steps,
            enable_nvml_sampling=enable_nvml_sampling,
            measure_serialized_model_size=measure_serialized_model_size,
        )

        self.enable_gradient_logging=bool(enable_gradient_logging)
        self.gradient_every_n_steps=max(1,int(gradient_every_n_steps))
        self.gradient_steps=None if gradient_steps is None else {int(x) for x in gradient_steps}
        self.gradient_log_dot_products=bool(gradient_log_dot_products)

        self.logs["gradient_telemetry"]=[]
        self.logs["config"].update({
            "enable_gradient_logging":self.enable_gradient_logging,
            "gradient_every_n_steps":self.gradient_every_n_steps,
            "gradient_steps":None if self.gradient_steps is None else tuple(sorted(self.gradient_steps)),
            "gradient_log_dot_products":self.gradient_log_dot_products,
        })

    # ==================================================================
    # Gradient helpers
    # ==================================================================

    def _should_log_gradients(self, completed_step):
        if not self.enable_gradient_logging:
            return False
        completed_step=int(completed_step)

        if self.gradient_steps is not None:
            return completed_step in self.gradient_steps

        return (
            completed_step == 1 or
            completed_step == self.steps or
            completed_step % self.gradient_every_n_steps == 0
        )

    @staticmethod
    def _component_gradients(loss, params):
        """
        Return detached gradients of `loss` w.r.t. all trainable parameters.

        This uses autograd.grad rather than backward(), so parameter .grad
        buffers are not modified.
        """
        if loss is None:
            return None

        grads=torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        return tuple(
            None if g is None else g.detach()
            for g in grads
        )

    @staticmethod
    def _gradient_summary(grads):
        """
        Aggregate one gradient vector without flattening the full model.

        Returns:
          l2
          l2_squared
          max_abs
          n_elements
          n_tensors
        """
        if grads is None:
            return {
                "l2":np.nan,
                "l2_squared":np.nan,
                "max_abs":np.nan,
                "n_elements":0,
                "n_tensors":0,
            }

        sq_terms=[]
        max_terms=[]
        n_elements=0
        n_tensors=0

        for g in grads:
            if g is None:
                continue

            gf=g.float()
            sq_terms.append(torch.sum(gf*gf))
            max_terms.append(torch.max(torch.abs(gf)))
            n_elements+=int(g.numel())
            n_tensors+=1

        if not sq_terms:
            return {
                "l2":0.0,
                "l2_squared":0.0,
                "max_abs":0.0,
                "n_elements":0,
                "n_tensors":0,
            }

        sq=torch.stack(sq_terms).sum()
        max_abs=torch.stack(max_terms).max()

        sq_v=float(sq.item())

        return {
            "l2":float(np.sqrt(sq_v)),
            "l2_squared":sq_v,
            "max_abs":float(max_abs.item()),
            "n_elements":int(n_elements),
            "n_tensors":int(n_tensors),
        }

    @staticmethod
    def _gradient_dot(grads_a, grads_b):
        """
        Exact dot product between two component gradients.

        Logging this plus each component norm is sufficient to compute
        cosine similarity later without storing full gradient vectors.
        """
        if grads_a is None or grads_b is None:
            return np.nan

        terms=[]

        for ga,gb in zip(grads_a,grads_b):
            if ga is None or gb is None:
                continue
            terms.append(torch.sum(ga.float()*gb.float()))

        if not terms:
            return 0.0

        return float(torch.stack(terms).sum().item())

    @staticmethod
    def _actual_parameter_grad_summary(params):
        """
        Summarize the gradients currently stored in parameter .grad fields.

        Must be called AFTER GradScaler.unscale_(optimizer) when AMP scaling
        is active.
        """
        grads=tuple(
            None if p.grad is None else p.grad.detach()
            for p in params
        )
        return PepCL_Logging_Ext_Grad_Norm._gradient_summary(grads)

    @staticmethod
    def _add_component_gradient_log(row, prefix, grads, weight):
        """
        Add raw and analytically weighted component-gradient summaries.

        Because each component weight is a scalar:
            ||∇(wL)|| = |w| ||∇L||
            max|∇(wL)| = |w| max|∇L|
        so a second autograd traversal for the weighted loss is unnecessary.
        """
        s=PepCL_Logging_Ext_Grad_Norm._gradient_summary(grads)
        w=float(weight)

        row[f"{prefix}_grad_norm_raw"]=s["l2"]
        row[f"{prefix}_grad_norm_raw_squared"]=s["l2_squared"]
        row[f"{prefix}_grad_max_abs_raw"]=s["max_abs"]
        row[f"{prefix}_grad_n_elements"]=s["n_elements"]
        row[f"{prefix}_grad_n_tensors"]=s["n_tensors"]

        row[f"{prefix}_weight"]=w
        row[f"{prefix}_grad_norm_weighted"]=abs(w)*s["l2"] if np.isfinite(s["l2"]) else np.nan
        row[f"{prefix}_grad_norm_weighted_squared"]=(w*w)*s["l2_squared"] if np.isfinite(s["l2_squared"]) else np.nan
        row[f"{prefix}_grad_max_abs_weighted"]=abs(w)*s["max_abs"] if np.isfinite(s["max_abs"]) else np.nan

    def get_logs(self, as_dataframes=False):
        out=super().get_logs(as_dataframes=as_dataframes)

        if as_dataframes:
            out["gradient_telemetry"]=pd.DataFrame(out.get("gradient_telemetry",[]))

        return out

    # ==================================================================
    # Training
    # ==================================================================

    def train(self):
        train_call_t0=time.perf_counter()
        self._snapshot("train_enter",detailed_memory=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # --------------------------------------------------------------
        # Sampling setup — identical behavior to PepCL_Logging_Ext.
        # --------------------------------------------------------------
        t=time.perf_counter()

        if not self.allele_aware_sampling:
            print(
                "[PepCL_Logging_Ext_Grad_Norm] Naive balanced sampling: "
                "allele-aware sampling disabled; each active stream samples "
                "1:1 positives/negatives globally. num_*_pos and *_neg_ratio "
                "are ignored.\n"
            )

        elif self.auto_calibrate:
            self._auto_calibrate()

        else:
            if (
                self.need_ft_stream and
                self.ft_is_continuous and
                self.continuous_ft_sampling_mode=="pooled"
            ):
                self.ft_cfg["sampling_summary"]=self._compute_pooled_ft_summary()

            print(
                f"[PepCL_Logging_Ext_Grad_Norm] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, "
                f"MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, "
                f"FT neg_ratio={self.ft_cfg['ratio']} | "
                f"FT continuous={self.ft_is_continuous}, "
                f"FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )

        self.logs["timing"]["sampling_setup_seconds"]=time.perf_counter()-t
        self._snapshot("after_sampling_setup",detailed_memory=True)

        # --------------------------------------------------------------
        # Loaders.
        # --------------------------------------------------------------
        ms_iter=ft_iter=None

        t=time.perf_counter()
        if self.need_ms_stream:
            ms_loader=self._build_loader(0,self.ms_cfg,self.seed)
            self.logs["timing"]["ms_loader_construct_seconds"]=time.perf_counter()-t
            self.logs["data_loader"]["ms"]=self._loader_stats(ms_loader)

            t_iter=time.perf_counter()
            ms_iter=iter(ms_loader)
            self.logs["timing"]["ms_loader_iterator_start_seconds"]=time.perf_counter()-t_iter

        else:
            self.logs["timing"]["ms_loader_construct_seconds"]=0.0
            self.logs["timing"]["ms_loader_iterator_start_seconds"]=0.0

        t=time.perf_counter()
        if self.need_ft_stream:
            ft_loader=self._build_loader(1,self.ft_cfg,self.seed+1)
            self.logs["timing"]["ft_loader_construct_seconds"]=time.perf_counter()-t
            self.logs["data_loader"]["ft"]=self._loader_stats(ft_loader)

            t_iter=time.perf_counter()
            ft_iter=iter(ft_loader)
            self.logs["timing"]["ft_loader_iterator_start_seconds"]=time.perf_counter()-t_iter

        else:
            self.logs["timing"]["ft_loader_construct_seconds"]=0.0
            self.logs["timing"]["ft_loader_iterator_start_seconds"]=0.0

        self._snapshot("after_loader_iterator_start",detailed_memory=True)

        # --------------------------------------------------------------
        # Optimizer / scheduler / AMP scaler.
        # --------------------------------------------------------------
        t=time.perf_counter()
        optimizer=self._get_optimizer()

        warmup_steps=int(self.steps*self.warmup_ratio)

        if self.lr_scheduler_type=="linear":
            scheduler=get_linear_schedule_with_warmup(
                optimizer,warmup_steps,self.steps
            )
        elif self.lr_scheduler_type=="warmup_constant":
            scheduler=get_constant_schedule_with_warmup(
                optimizer,warmup_steps
            )
        else:
            scheduler=get_constant_schedule(optimizer)

        scaler=GradScaler()

        self.logs["timing"]["optimizer_scheduler_scaler_setup_seconds"]=time.perf_counter()-t
        self._snapshot(
            "after_optimizer_setup_before_state_allocation",
            detailed_memory=True
        )

        metrics=collections.defaultdict(list)

        model_checkpoints=(
            {}
            if self.save_every_n_steps is not None and self.output_dir is None
            else None
        )

        pbar=tqdm(
            range(self.steps),
            desc="PepCL_Logging_Ext_Grad_Norm"
        )

        compute=self.logs["compute"]
        train_loop_t0=time.perf_counter()
        first_optimizer_snapshot_done=False
        step_wall_times=[]

        # Exact parameter set used for component-gradient diagnostics.
        trainable_params=[
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        self.logs["config"]["gradient_trainable_parameter_tensors"]=len(trainable_params)
        self.logs["config"]["gradient_trainable_parameter_elements"]=int(
            sum(p.numel() for p in trainable_params)
        )

        # ==============================================================
        # Training loop
        # ==============================================================
        for step in pbar:
            completed_step=step+1
            log_grads=self._should_log_gradients(completed_step)

            step_t0=time.perf_counter()

            self.model.train()
            optimizer.zero_grad()

            # Legacy values.
            l_ms_v=l_ft_v=l_dr_v=0.0

            # Loss tensors.
            loss_ms_cls_raw=loss_ms_cls_weighted=None
            loss_ft_raw=loss_ft_weighted=None
            loss_dr_raw=loss_dr_weighted=None
            loss_ms=loss_ft=total_l=None

            # Component gradients exist only on diagnostic steps.
            g_replay=g_dr=g_ft=None

            grad_row={
                "step":int(completed_step),
                "ft_is_continuous":bool(self.ft_is_continuous),
                "ms_loss_type":self.ms_loss_type,
                "ft_loss_type":self.nonms_loss_type,
                "dr_loss_type":self.dr_loss_type,
            } if log_grads else None

            grad_diag_t0=time.perf_counter() if log_grads else None

            # ==========================================================
            # Domain 0: Replay/MS classification + DR
            # ==========================================================
            if self.need_ms_stream:
                try:
                    t_fetch=time.perf_counter()
                    batch_ms=next(ms_iter)
                    compute["ms_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

                    t_move=time.perf_counter()
                    batch_ms=self._move_batch(batch_ms)
                    compute["ms_host_to_device_seconds"]+=time.perf_counter()-t_move

                    lbl_ms=batch_ms.pop("label")
                    ms_n=int(lbl_ms.numel())

                    compute["ms_batches"]+=1
                    compute["ms_samples"]+=ms_n

                    with autocast(self.device.type):
                        logits_ms=self.model(batch_ms).view(-1)

                        compute["student_forward_passes_ms"]+=1
                        compute["student_forward_samples_ms"]+=ms_n

                        if self.compute_ms_classification:
                            loss_ms_cls_raw=self.criterion_ms(
                                logits_ms,lbl_ms
                            )
                            loss_ms_cls_weighted=(
                                loss_ms_cls_raw*self.ms_lambda
                            )

                        if self.dr_lambda>0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits=self.teacher(batch_ms).view(-1)

                            compute["teacher_forward_passes"]+=1
                            compute["teacher_forward_samples"]+=ms_n

                            loss_dr_raw=self._get_dr_loss(
                                logits_ms,t_logits
                            )
                            loss_dr_weighted=(
                                loss_dr_raw*self.dr_lambda
                            )

                        if (
                            loss_ms_cls_weighted is not None and
                            loss_dr_weighted is not None
                        ):
                            loss_ms=(
                                loss_ms_cls_weighted+
                                loss_dr_weighted
                            )

                        elif loss_ms_cls_weighted is not None:
                            loss_ms=loss_ms_cls_weighted

                        else:
                            loss_ms=loss_dr_weighted

                    # --------------------------------------------------
                    # Isolated component gradients.
                    # --------------------------------------------------
                    if log_grads:
                        if loss_ms_cls_raw is not None:
                            g_replay=self._component_gradients(
                                loss_ms_cls_raw,
                                trainable_params,
                            )

                            self._add_component_gradient_log(
                                grad_row,
                                "replay",
                                g_replay,
                                self.ms_lambda,
                            )

                        else:
                            self._add_component_gradient_log(
                                grad_row,
                                "replay",
                                None,
                                self.ms_lambda,
                            )

                        if loss_dr_raw is not None:
                            g_dr=self._component_gradients(
                                loss_dr_raw,
                                trainable_params,
                            )

                            self._add_component_gradient_log(
                                grad_row,
                                "dr",
                                g_dr,
                                self.dr_lambda,
                            )

                        else:
                            self._add_component_gradient_log(
                                grad_row,
                                "dr",
                                None,
                                self.dr_lambda,
                            )

                        if self.gradient_log_dot_products:
                            raw_dot=self._gradient_dot(
                                g_replay,g_dr
                            )

                            grad_row["replay_dr_dot_raw"]=raw_dot
                            grad_row["replay_dr_dot_weighted"]=(
                                raw_dot*self.ms_lambda*self.dr_lambda
                                if np.isfinite(raw_dot)
                                else np.nan
                            )

                    # Normal PepCL training backward.
                    if self.seq_backprop and loss_ms is not None:
                        scaler.scale(loss_ms).backward()
                        compute["backward_passes"]+=1

                except (StopIteration,AttributeError):
                    pass

            # ==========================================================
            # Domain 1: FT/update objective
            # ==========================================================
            if self.need_ft_stream:
                try:
                    t_fetch=time.perf_counter()
                    batch_ft=next(ft_iter)
                    compute["ft_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

                    t_move=time.perf_counter()
                    batch_ft=self._move_batch(batch_ft)
                    compute["ft_host_to_device_seconds"]+=time.perf_counter()-t_move

                    lbl_ft=batch_ft.pop("label")
                    ft_n=int(lbl_ft.numel())

                    compute["ft_batches"]+=1
                    compute["ft_samples"]+=ft_n

                    with autocast(self.device.type):
                        logits_ft=self.model(batch_ft).view(-1)

                        compute["student_forward_passes_ft"]+=1
                        compute["student_forward_samples_ft"]+=ft_n

                        loss_ft_raw=self.criterion_ft(
                            logits_ft,lbl_ft
                        )
                        loss_ft_weighted=(
                            loss_ft_raw*self.ft_lambda
                        )
                        loss_ft=loss_ft_weighted

                    # --------------------------------------------------
                    # Isolated FT gradient.
                    # --------------------------------------------------
                    if log_grads:
                        g_ft=self._component_gradients(
                            loss_ft_raw,
                            trainable_params,
                        )

                        self._add_component_gradient_log(
                            grad_row,
                            "ft",
                            g_ft,
                            self.ft_lambda,
                        )

                        if self.gradient_log_dot_products:
                            replay_ft_dot=self._gradient_dot(
                                g_replay,g_ft
                            )
                            dr_ft_dot=self._gradient_dot(
                                g_dr,g_ft
                            )

                            grad_row["replay_ft_dot_raw"]=replay_ft_dot
                            grad_row["replay_ft_dot_weighted"]=(
                                replay_ft_dot*self.ms_lambda*self.ft_lambda
                                if np.isfinite(replay_ft_dot)
                                else np.nan
                            )

                            grad_row["dr_ft_dot_raw"]=dr_ft_dot
                            grad_row["dr_ft_dot_weighted"]=(
                                dr_ft_dot*self.dr_lambda*self.ft_lambda
                                if np.isfinite(dr_ft_dot)
                                else np.nan
                            )

                    # Normal PepCL training backward.
                    if self.seq_backprop and loss_ft is not None:
                        scaler.scale(loss_ft).backward()
                        compute["backward_passes"]+=1

                except (StopIteration,AttributeError):
                    pass

            # FT inactive.
            elif log_grads:
                self._add_component_gradient_log(
                    grad_row,
                    "ft",
                    None,
                    self.ft_lambda,
                )

            # ==========================================================
            # Total optimization objective
            # ==========================================================
            if loss_ms is not None:
                total_l=loss_ms

            if loss_ft is not None:
                total_l=(
                    loss_ft
                    if total_l is None
                    else total_l+loss_ft
                )

            if not self.seq_backprop and total_l is not None:
                scaler.scale(total_l).backward()
                compute["backward_passes"]+=1

            # ----------------------------------------------------------
            # Actual accumulated gradient used by optimizer.
            #
            # Unscale ONCE before inspecting .grad. scaler.step() knows
            # unscale_ has already occurred and will not do it twice.
            # ----------------------------------------------------------
            if log_grads and total_l is not None:
                scaler.unscale_(optimizer)

                total_grad=self._actual_parameter_grad_summary(
                    trainable_params
                )

                grad_row["total_grad_norm_actual"]=total_grad["l2"]
                grad_row["total_grad_norm_actual_squared"]=total_grad["l2_squared"]
                grad_row["total_grad_max_abs_actual"]=total_grad["max_abs"]
                grad_row["total_grad_n_elements"]=total_grad["n_elements"]
                grad_row["total_grad_n_tensors"]=total_grad["n_tensors"]

            # Actual optimizer update.
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            compute["optimizer_steps"]+=1
            compute["scheduler_steps"]+=1

            if not first_optimizer_snapshot_done:
                self._snapshot(
                    "after_first_optimizer_step_state_allocated",
                    step=1,
                    detailed_memory=True,
                )
                first_optimizer_snapshot_done=True

            # ----------------------------------------------------------
            # Scalar loss values.
            # ----------------------------------------------------------
            def _loss_value(x):
                return (
                    float(x.detach().item())
                    if x is not None
                    else np.nan
                )

            ms_cls_raw_v=_loss_value(loss_ms_cls_raw)
            ms_cls_weighted_v=_loss_value(loss_ms_cls_weighted)

            ft_raw_v=_loss_value(loss_ft_raw)
            ft_weighted_v=_loss_value(loss_ft_weighted)

            dr_raw_v=_loss_value(loss_dr_raw)
            dr_weighted_v=_loss_value(loss_dr_weighted)

            total_loss_v=_loss_value(total_l)

            if self.compute_ms_classification and loss_ms is not None:
                l_ms_v=_loss_value(loss_ms)

            if loss_ft_weighted is not None:
                l_ft_v=ft_weighted_v

            if loss_dr_weighted is not None:
                l_dr_v=dr_weighted_v

            pbar.set_postfix({
                "MS":f"{l_ms_v:.6f}",
                "FT":f"{l_ft_v:.6f}",
                "DR":f"{l_dr_v:.6f}",
            })

            # ----------------------------------------------------------
            # Existing loss trajectories.
            # ----------------------------------------------------------
            metrics["ms_loss"].append(l_ms_v)
            metrics["ft_loss"].append(l_ft_v)
            metrics["dr_loss"].append(l_dr_v)

            metrics["ms_classification_loss_raw"].append(
                ms_cls_raw_v
            )
            metrics["ms_classification_loss_weighted"].append(
                ms_cls_weighted_v
            )

            metrics["ft_loss_raw"].append(
                ft_raw_v
            )
            metrics["ft_loss_weighted"].append(
                ft_weighted_v
            )

            metrics["drift_loss_raw"].append(
                dr_raw_v
            )
            metrics["drift_loss_weighted"].append(
                dr_weighted_v
            )

            metrics["total_loss"].append(
                total_loss_v
            )

            metrics["continuous_loss_raw"].append(
                ft_raw_v
                if self.ft_is_continuous
                else np.nan
            )
            metrics["continuous_loss_weighted"].append(
                ft_weighted_v
                if self.ft_is_continuous
                else np.nan
            )

            # ----------------------------------------------------------
            # Finalize gradient row.
            # ----------------------------------------------------------
            if log_grads:
                grad_row.update({
                    "ms_classification_loss_raw":ms_cls_raw_v,
                    "ms_classification_loss_weighted":ms_cls_weighted_v,
                    "ft_loss_raw":ft_raw_v,
                    "ft_loss_weighted":ft_weighted_v,
                    "drift_loss_raw":dr_raw_v,
                    "drift_loss_weighted":dr_weighted_v,
                    "total_loss":total_loss_v,
                    "gradient_diagnostic_seconds":time.perf_counter()-grad_diag_t0,
                })

                self.logs["gradient_telemetry"].append(
                    grad_row
                )

                compute["gradient_diagnostic_steps"]+=1
                compute["gradient_diagnostic_seconds"]+=grad_row[
                    "gradient_diagnostic_seconds"
                ]

            # Component gradient tensors no longer needed.
            del g_replay,g_dr,g_ft

            # ----------------------------------------------------------
            # Checkpoint saving.
            # ----------------------------------------------------------
            if self.save_every_n_steps is None:
                should_save=False

            elif isinstance(self.save_every_n_steps,int):
                should_save=(
                    completed_step%
                    self.save_every_n_steps==0
                )

            else:
                should_save=(
                    completed_step in
                    self.save_every_n_steps
                )

            if should_save:
                t_save=time.perf_counter()

                if self.output_dir is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.output_dir,
                            f"pepcl_step_{completed_step}.pt"
                        )
                    )

                else:
                    model_checkpoints[completed_step]=(
                        copy.deepcopy(self.model)
                        .cpu()
                        .eval()
                    )

                compute["checkpoint_count"]+=1
                compute["checkpoint_seconds"]+=(
                    time.perf_counter()-t_save
                )

                self._snapshot(
                    f"after_checkpoint_{completed_step}",
                    step=completed_step,
                )

            step_wall_s=time.perf_counter()-step_t0
            step_wall_times.append(step_wall_s)

            compute["sum_step_wall_seconds"]+=step_wall_s

            self._telemetry(
                completed_step,
                step_wall_s,
                completed_step,
                train_loop_t0,
            )

        # ==============================================================
        # End-of-run logging — same structure as parent class.
        # ==============================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        train_loop_seconds=(
            time.perf_counter()-train_loop_t0
        )

        self.logs["timing"]["training_loop_seconds"]=(
            train_loop_seconds
        )

        self.logs["timing"]["train_call_total_seconds"]=(
            time.perf_counter()-train_call_t0
        )

        self.logs["timing"]["end_to_end_seconds"]=(
            time.perf_counter()-self._object_t0
        )

        self.logs["timing"]["average_steps_per_second"]=(
            self.steps/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        self.logs["timing"]["average_seconds_per_step"]=(
            train_loop_seconds/self.steps
            if self.steps>0
            else np.nan
        )

        self.logs["timing"]["first_step_seconds"]=(
            float(step_wall_times[0])
            if step_wall_times
            else np.nan
        )

        if len(step_wall_times)>10:
            steady_s=float(sum(step_wall_times[10:]))
            steady_n=len(step_wall_times)-10

            self.logs["timing"]["steady_state_start_step"]=11

            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=(
                steady_n/steady_s
                if steady_s>0
                else np.nan
            )

            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=(
                steady_s/steady_n
                if steady_n>0
                else np.nan
            )

        else:
            self.logs["timing"]["steady_state_start_step"]=None
            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=np.nan
            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=np.nan

        compute["student_forward_passes_total"]=(
            compute["student_forward_passes_ms"]+
            compute["student_forward_passes_ft"]
        )

        compute["student_forward_samples_total"]=(
            compute["student_forward_samples_ms"]+
            compute["student_forward_samples_ft"]
        )

        compute["total_forward_passes_including_teacher"]=(
            compute["student_forward_passes_total"]+
            compute["teacher_forward_passes"]
        )

        compute["total_forward_samples_including_teacher"]=(
            compute["student_forward_samples_total"]+
            compute["teacher_forward_samples"]
        )

        compute["student_samples_per_second"]=(
            compute["student_forward_samples_total"]/
            train_loop_seconds
            if compute["student_forward_samples_total"]>0
            else np.nan
        )

        compute["ms_samples_per_second"]=(
            compute["student_forward_samples_ms"]/
            train_loop_seconds
            if compute["student_forward_samples_ms"]>0
            else np.nan
        )

        compute["ft_samples_per_second"]=(
            compute["student_forward_samples_ft"]/
            train_loop_seconds
            if compute["student_forward_samples_ft"]>0
            else np.nan
        )

        compute["teacher_samples_per_second"]=(
            compute["teacher_forward_samples"]/
            train_loop_seconds
            if compute["teacher_forward_samples"]>0
            else np.nan
        )

        compute["seconds_per_1000_student_samples"]=(
            1000.0*train_loop_seconds/
            compute["student_forward_samples_total"]
            if compute["student_forward_samples_total"]>0
            else np.nan
        )

        self._snapshot(
            "train_exit",
            step=self.steps,
            detailed_memory=True,
        )

        if torch.cuda.is_available():
            self.logs["model"]["train_peak_gpu_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device)
            )
            self.logs["model"]["train_peak_gpu_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device)
            )

            try:
                self.logs["cuda_allocator_stats_end"]={
                    k:int(v)
                    for k,v in
                    torch.cuda.memory_stats(self.device).items()
                }
            except Exception:
                self.logs["cuda_allocator_stats_end"]={}

        # Peak host-memory deltas.
        snaps=self.logs.get("memory_snapshots",[])
        baseline=self.logs.get("memory_baseline",{})

        if snaps:
            def _peak_delta(field):
                vals=[
                    x.get(field)
                    for x in snaps
                    if x.get(field) is not None
                ]
                b=baseline.get(field)

                return (
                    max(vals)-b
                    if vals and b is not None
                    else None
                )

            self.logs["model"]["peak_system_ram_delta_gb"]=(
                _peak_delta("system_ram_used_gb")
            )
            self.logs["model"]["peak_process_rss_delta_gb"]=(
                _peak_delta("process_rss_gb")
            )
            self.logs["model"]["peak_process_tree_rss_delta_gb"]=(
                _peak_delta("process_tree_rss_gb")
            )
            self.logs["model"]["peak_process_uss_delta_gb"]=(
                _peak_delta("process_uss_gb")
            )
            self.logs["model"]["peak_process_pss_delta_gb"]=(
                _peak_delta("process_pss_gb")
            )
            self.logs["model"]["peak_process_tree_uss_delta_gb"]=(
                _peak_delta("process_tree_uss_gb")
            )
            self.logs["model"]["peak_process_tree_pss_delta_gb"]=(
                _peak_delta("process_tree_pss_gb")
            )

        self.logs["notes"].append(
            "Gradient diagnostics use isolated torch.autograd.grad calls on "
            "the raw replay, DR, and FT objectives. These calls do not "
            "accumulate into parameter .grad and therefore do not alter the "
            "optimization objective. Pairwise raw/weighted gradient dot "
            "products are logged so cosine similarities can be computed "
            "post hoc without retaining full gradient vectors."
        )

        self.logs["notes"].append(
            "Gradient diagnostics increase runtime because each sampled "
            "diagnostic step requires additional backward-like autograd "
            "traversals. Use gradient_every_n_steps or gradient_steps to "
            "control diagnostic frequency."
        )

        if model_checkpoints is not None:
            return metrics,model_checkpoints

        return metrics
    

class PepCL_Logging_Ext_Early_Stopping(PepCL_Logging_Ext):
    """
    PepCL_Logging_Ext with fixed-step early stopping.

    IMPORTANT:
      `steps` remains the FULL training budget and therefore defines the
      scheduler/warmup exactly as in a normal full run.

      `early_stopping_step` only truncates execution.

    Example:
        steps=800, early_stopping_step=200

    is intended to be equivalent to:
        normal steps=800 training -> use checkpoint at step 200

    NOT equivalent to:
        steps=200
    because that would change the LR scheduler/warmup horizon.
    """

    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,
        mhc_seq_dict: dict,

        steps: int = 200,
        early_stopping_step: int = None,
        output_dir: str = None,
        save_every_n_steps=None,
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
        allele_aware_sampling: bool = True,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
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
        dr_loss_type: str = "mse",

        compact_dataframe: bool = True,
        compact_dtypes: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,

        enable_logging: bool = True,
        telemetry_every_n_steps: int = 10,
        enable_nvml_sampling: bool = True,
        measure_serialized_model_size: bool = False,
    ):
        super().__init__(
            model=model, train_df=train_df, tokenizer=tokenizer, device=device, mhc_seq_dict=mhc_seq_dict,
            steps=steps, output_dir=output_dir, save_every_n_steps=save_every_n_steps,
            seed=seed, num_workers=num_workers, collate_fn=collate_fn,

            num_ms_pos=num_ms_pos, ms_neg_ratio=ms_neg_ratio,
            num_nonms_pos=num_nonms_pos, nonms_neg_ratio=nonms_neg_ratio,
            ms_batch_size=ms_batch_size, nonms_batch_size=nonms_batch_size,

            auto_calibrate=auto_calibrate, calibration_steps=calibration_steps,
            continuous_target_col=continuous_target_col,
            continuous_ft_sampling_mode=continuous_ft_sampling_mode,
            pooled_ft_num_pos=pooled_ft_num_pos,
            allele_aware_sampling=allele_aware_sampling,

            encoder_lr=encoder_lr, decoder_lr=decoder_lr,
            weight_decay=weight_decay, warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            update_domain_embeddings=update_domain_embeddings,

            use_replay=use_replay, primary_task_id=primary_task_id,
            sequential_backprop=sequential_backprop,

            ms_loss_type=ms_loss_type, ms_loss_alpha=ms_loss_alpha, ms_lambda=ms_lambda,
            nonms_loss_type=nonms_loss_type, nonms_loss_alpha=nonms_loss_alpha,
            ft_lambda=ft_lambda, dr_lambda=dr_lambda, dr_loss_type=dr_loss_type,

            compact_dataframe=compact_dataframe, compact_dtypes=compact_dtypes,
            pin_memory=pin_memory, persistent_workers=persistent_workers,

            enable_logging=enable_logging, telemetry_every_n_steps=telemetry_every_n_steps,
            enable_nvml_sampling=enable_nvml_sampling,
            measure_serialized_model_size=measure_serialized_model_size,
        )

        if early_stopping_step is None:
            self.early_stopping_step=self.steps
        else:
            if not isinstance(early_stopping_step,(int,np.integer)):
                raise TypeError("early_stopping_step must be None or an integer.")
            if int(early_stopping_step)<=0:
                raise ValueError("early_stopping_step must be > 0.")
            if int(early_stopping_step)>self.steps:
                raise ValueError(
                    f"early_stopping_step ({early_stopping_step}) cannot exceed "
                    f"the full step budget ({self.steps})."
                )
            self.early_stopping_step=int(early_stopping_step)

        self.full_step_budget=int(self.steps)
        self.logs["config"]["full_step_budget"]=self.full_step_budget
        self.logs["config"]["early_stopping_step"]=self.early_stopping_step
        self.logs["config"]["actual_training_steps"]=self.early_stopping_step
        self.logs["notes"].append(
            "Early stopping truncates execution only. The LR scheduler and warmup "
            "remain parameterized by the full requested step budget, so stopping at "
            "step N matches taking step-N from the corresponding full-budget run."
        )

    def _telemetry(self, step, step_wall_s, completed_steps, train_loop_t0):
        if not self.enable_logging: return
        terminal_step=self.early_stopping_step
        if step!=1 and step%self.telemetry_every_n_steps!=0 and step!=terminal_step: return
        elapsed=time.perf_counter()-train_loop_t0
        row={
            "step":int(step),"step_wall_seconds":float(step_wall_s),
            "cumulative_train_seconds":float(elapsed),
            "cumulative_steps_per_second":float(completed_steps/elapsed) if elapsed>0 else np.nan,
        }
        self._snapshot(f"train_step_{step}",step=step,detailed_memory=False)
        if self.logs["memory_snapshots"]:
            row.update({k:v for k,v in self.logs["memory_snapshots"][-1].items() if k not in {"tag","step"}})
        self.logs["step_telemetry"].append(row)

    def train(self):
        train_call_t0=time.perf_counter()
        self._snapshot("train_enter",detailed_memory=True)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(self.device)

        # Sampling setup -- unchanged.
        t=time.perf_counter()
        if not self.allele_aware_sampling:
            print(
                "[PepCL_Logging_Ext_Early_Stopping] Naive balanced sampling: allele-aware sampling disabled; "
                "each active stream samples 1:1 positives/negatives globally. num_*_pos and *_neg_ratio are ignored.\n"
            )
        elif self.auto_calibrate:
            self._auto_calibrate()
        else:
            if self.need_ft_stream and self.ft_is_continuous and self.continuous_ft_sampling_mode=="pooled":
                self.ft_cfg["sampling_summary"]=self._compute_pooled_ft_summary()
            print(
                f"[PepCL_Logging_Ext_Early_Stopping] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, FT neg_ratio={self.ft_cfg['ratio']} | "
                f"FT continuous={self.ft_is_continuous}, FT sampling_mode={self.ft_cfg['sampling_mode']}\n"
            )

        print(
            f"[PepCL_Logging_Ext_Early_Stopping] Full step budget={self.full_step_budget} | "
            f"stop after step={self.early_stopping_step} | "
            f"scheduler remains configured for {self.full_step_budget} steps.\n"
        )

        self.logs["timing"]["sampling_setup_seconds"]=time.perf_counter()-t
        self._snapshot("after_sampling_setup",detailed_memory=True)

        # Loaders -- unchanged.
        ms_iter=ft_iter=None

        t=time.perf_counter()
        if self.need_ms_stream:
            ms_loader=self._build_loader(0,self.ms_cfg,self.seed)
            self.logs["timing"]["ms_loader_construct_seconds"]=time.perf_counter()-t
            self.logs["data_loader"]["ms"]=self._loader_stats(ms_loader)
            t_iter=time.perf_counter(); ms_iter=iter(ms_loader)
            self.logs["timing"]["ms_loader_iterator_start_seconds"]=time.perf_counter()-t_iter
        else:
            self.logs["timing"]["ms_loader_construct_seconds"]=0.0
            self.logs["timing"]["ms_loader_iterator_start_seconds"]=0.0

        t=time.perf_counter()
        if self.need_ft_stream:
            ft_loader=self._build_loader(1,self.ft_cfg,self.seed+1)
            self.logs["timing"]["ft_loader_construct_seconds"]=time.perf_counter()-t
            self.logs["data_loader"]["ft"]=self._loader_stats(ft_loader)
            t_iter=time.perf_counter(); ft_iter=iter(ft_loader)
            self.logs["timing"]["ft_loader_iterator_start_seconds"]=time.perf_counter()-t_iter
        else:
            self.logs["timing"]["ft_loader_construct_seconds"]=0.0
            self.logs["timing"]["ft_loader_iterator_start_seconds"]=0.0

        self._snapshot("after_loader_iterator_start",detailed_memory=True)

        # IMPORTANT: scheduler uses FULL self.steps, not early_stopping_step.
        t=time.perf_counter()
        optimizer=self._get_optimizer()
        warmup_steps=int(self.steps*self.warmup_ratio)

        if self.lr_scheduler_type=="linear":
            scheduler=get_linear_schedule_with_warmup(optimizer,warmup_steps,self.steps)
        elif self.lr_scheduler_type=="warmup_constant":
            scheduler=get_constant_schedule_with_warmup(optimizer,warmup_steps)
        else:
            scheduler=get_constant_schedule(optimizer)

        scaler=GradScaler()
        self.logs["timing"]["optimizer_scheduler_scaler_setup_seconds"]=time.perf_counter()-t
        self._snapshot("after_optimizer_setup_before_state_allocation",detailed_memory=True)

        metrics=collections.defaultdict(list)
        model_checkpoints={} if self.save_every_n_steps is not None and self.output_dir is None else None

        # ONLY substantive change: execute early_stopping_step steps rather than self.steps.
        pbar=tqdm(range(self.early_stopping_step),desc="PepCL_Logging_Ext_Early_Stopping")

        compute=self.logs["compute"]
        train_loop_t0=time.perf_counter()
        first_optimizer_snapshot_done=False
        step_wall_times=[]

        for step in pbar:
            step_t0=time.perf_counter()
            self.model.train()
            optimizer.zero_grad()

            l_ms_v=l_ft_v=l_dr_v=0.0
            loss_ms_cls_raw=loss_ms_cls_weighted=None
            loss_ft_raw=loss_ft_weighted=None
            loss_dr_raw=loss_dr_weighted=None
            loss_ms=loss_ft=total_l=None

            # Domain 0 / replay and/or drift regularization -- unchanged.
            if self.need_ms_stream:
                try:
                    t_fetch=time.perf_counter(); batch_ms=next(ms_iter)
                    compute["ms_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

                    t_move=time.perf_counter(); batch_ms=self._move_batch(batch_ms)
                    compute["ms_host_to_device_seconds"]+=time.perf_counter()-t_move

                    lbl_ms=batch_ms.pop("label")
                    ms_n=int(lbl_ms.numel())
                    compute["ms_batches"]+=1
                    compute["ms_samples"]+=ms_n

                    with autocast(self.device.type):
                        logits_ms=self.model(batch_ms).view(-1)
                        compute["student_forward_passes_ms"]+=1
                        compute["student_forward_samples_ms"]+=ms_n

                        if self.compute_ms_classification:
                            loss_ms_cls_raw=self.criterion_ms(logits_ms,lbl_ms)
                            loss_ms_cls_weighted=loss_ms_cls_raw*self.ms_lambda

                        if self.dr_lambda>0 and self.teacher is not None:
                            with torch.no_grad():
                                t_logits=self.teacher(batch_ms).view(-1)
                            compute["teacher_forward_passes"]+=1
                            compute["teacher_forward_samples"]+=ms_n
                            loss_dr_raw=self._get_dr_loss(logits_ms,t_logits)
                            loss_dr_weighted=loss_dr_raw*self.dr_lambda

                        if loss_ms_cls_weighted is not None and loss_dr_weighted is not None:
                            loss_ms=loss_ms_cls_weighted+loss_dr_weighted
                        elif loss_ms_cls_weighted is not None:
                            loss_ms=loss_ms_cls_weighted
                        else:
                            loss_ms=loss_dr_weighted

                    if self.seq_backprop and loss_ms is not None:
                        scaler.scale(loss_ms).backward()
                        compute["backward_passes"]+=1

                except (StopIteration,AttributeError):
                    pass

            # Domain 1 / FT/update stream -- unchanged.
            if self.need_ft_stream:
                try:
                    t_fetch=time.perf_counter(); batch_ft=next(ft_iter)
                    compute["ft_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

                    t_move=time.perf_counter(); batch_ft=self._move_batch(batch_ft)
                    compute["ft_host_to_device_seconds"]+=time.perf_counter()-t_move

                    lbl_ft=batch_ft.pop("label")
                    ft_n=int(lbl_ft.numel())
                    compute["ft_batches"]+=1
                    compute["ft_samples"]+=ft_n

                    with autocast(self.device.type):
                        logits_ft=self.model(batch_ft).view(-1)
                        compute["student_forward_passes_ft"]+=1
                        compute["student_forward_samples_ft"]+=ft_n

                        loss_ft_raw=self.criterion_ft(logits_ft,lbl_ft)
                        loss_ft_weighted=loss_ft_raw*self.ft_lambda
                        loss_ft=loss_ft_weighted

                    if self.seq_backprop and loss_ft is not None:
                        scaler.scale(loss_ft).backward()
                        compute["backward_passes"]+=1

                except (StopIteration,AttributeError):
                    pass

            if loss_ms is not None: total_l=loss_ms
            if loss_ft is not None: total_l=loss_ft if total_l is None else total_l+loss_ft

            if not self.seq_backprop and total_l is not None:
                scaler.scale(total_l).backward()
                compute["backward_passes"]+=1

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            compute["optimizer_steps"]+=1
            compute["scheduler_steps"]+=1

            if not first_optimizer_snapshot_done:
                self._snapshot("after_first_optimizer_step_state_allocated",step=1,detailed_memory=True)
                first_optimizer_snapshot_done=True

            def _loss_value(x):
                return float(x.detach().item()) if x is not None else np.nan

            ms_cls_raw_v=_loss_value(loss_ms_cls_raw)
            ms_cls_weighted_v=_loss_value(loss_ms_cls_weighted)
            ft_raw_v=_loss_value(loss_ft_raw)
            ft_weighted_v=_loss_value(loss_ft_weighted)
            dr_raw_v=_loss_value(loss_dr_raw)
            dr_weighted_v=_loss_value(loss_dr_weighted)
            total_loss_v=_loss_value(total_l)

            if self.compute_ms_classification and loss_ms is not None: l_ms_v=_loss_value(loss_ms)
            if loss_ft_weighted is not None: l_ft_v=ft_weighted_v
            if loss_dr_weighted is not None: l_dr_v=dr_weighted_v

            pbar.set_postfix({"MS":f"{l_ms_v:.6f}","FT":f"{l_ft_v:.6f}","DR":f"{l_dr_v:.6f}"})

            metrics["ms_loss"].append(l_ms_v)
            metrics["ft_loss"].append(l_ft_v)
            metrics["dr_loss"].append(l_dr_v)

            metrics["ms_classification_loss_raw"].append(ms_cls_raw_v)
            metrics["ms_classification_loss_weighted"].append(ms_cls_weighted_v)
            metrics["ft_loss_raw"].append(ft_raw_v)
            metrics["ft_loss_weighted"].append(ft_weighted_v)
            metrics["drift_loss_raw"].append(dr_raw_v)
            metrics["drift_loss_weighted"].append(dr_weighted_v)
            metrics["total_loss"].append(total_loss_v)

            metrics["continuous_loss_raw"].append(ft_raw_v if self.ft_is_continuous else np.nan)
            metrics["continuous_loss_weighted"].append(ft_weighted_v if self.ft_is_continuous else np.nan)

            completed_step=step+1

            # Existing checkpoint behavior is unchanged.
            if self.save_every_n_steps is None:
                should_save=False
            elif isinstance(self.save_every_n_steps,int):
                should_save=completed_step%self.save_every_n_steps==0
            else:
                should_save=completed_step in self.save_every_n_steps

            if should_save:
                t_save=time.perf_counter()

                if self.output_dir is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir,f"pepcl_step_{completed_step}.pt")
                    )
                else:
                    model_checkpoints[completed_step]=copy.deepcopy(self.model).cpu().eval()

                compute["checkpoint_count"]+=1
                compute["checkpoint_seconds"]+=time.perf_counter()-t_save
                self._snapshot(f"after_checkpoint_{completed_step}",step=completed_step)

            step_wall_s=time.perf_counter()-step_t0
            step_wall_times.append(step_wall_s)
            compute["sum_step_wall_seconds"]+=step_wall_s
            self._telemetry(completed_step,step_wall_s,completed_step,train_loop_t0)

        # End-of-run logging. Use ACTUAL executed steps for timing statistics.
        if torch.cuda.is_available(): torch.cuda.synchronize(self.device)

        executed_steps=self.early_stopping_step
        train_loop_seconds=time.perf_counter()-train_loop_t0

        self.logs["timing"]["training_loop_seconds"]=train_loop_seconds
        self.logs["timing"]["train_call_total_seconds"]=time.perf_counter()-train_call_t0
        self.logs["timing"]["end_to_end_seconds"]=time.perf_counter()-self._object_t0
        self.logs["timing"]["average_steps_per_second"]=executed_steps/train_loop_seconds if train_loop_seconds>0 else np.nan
        self.logs["timing"]["average_seconds_per_step"]=train_loop_seconds/executed_steps if executed_steps>0 else np.nan
        self.logs["timing"]["first_step_seconds"]=float(step_wall_times[0]) if step_wall_times else np.nan

        self.logs["timing"]["requested_full_step_budget"]=self.full_step_budget
        self.logs["timing"]["executed_training_steps"]=executed_steps
        self.logs["timing"]["stopped_early"]=bool(executed_steps<self.full_step_budget)

        if len(step_wall_times)>10:
            steady_s=float(sum(step_wall_times[10:]))
            steady_n=len(step_wall_times)-10
            self.logs["timing"]["steady_state_start_step"]=11
            self.logs["timing"]["steady_state_steps_per_second_after_step_10"]=steady_n/steady_s if steady_s>0 else np.nan
            self.logs["timing"]["steady_state_seconds_per_step_after_step_10"]=steady_s/steady_n if steady_n>0 else np.nan
        else:
            self.logs["timing"]["steady_state_start_step"]=None
            self.logs["timing"]["steady_state_steps_per_second_after_step_10"]=np.nan
            self.logs["timing"]["steady_state_seconds_per_step_after_step_10"]=np.nan

        compute["student_forward_passes_total"]=compute["student_forward_passes_ms"]+compute["student_forward_passes_ft"]
        compute["student_forward_samples_total"]=compute["student_forward_samples_ms"]+compute["student_forward_samples_ft"]
        compute["total_forward_passes_including_teacher"]=compute["student_forward_passes_total"]+compute["teacher_forward_passes"]
        compute["total_forward_samples_including_teacher"]=compute["student_forward_samples_total"]+compute["teacher_forward_samples"]

        compute["student_samples_per_second"]=(
            compute["student_forward_samples_total"]/train_loop_seconds
            if train_loop_seconds>0 else np.nan
        )
        compute["ms_samples_per_second"]=(
            compute["student_forward_samples_ms"]/train_loop_seconds
            if train_loop_seconds>0 else np.nan
        )
        compute["ft_samples_per_second"]=(
            compute["student_forward_samples_ft"]/train_loop_seconds
            if train_loop_seconds>0 else np.nan
        )
        compute["teacher_samples_per_second"]=(
            compute["teacher_forward_samples"]/train_loop_seconds
            if train_loop_seconds>0 else np.nan
        )
        compute["seconds_per_1000_student_samples"]=(
            1000.0*train_loop_seconds/compute["student_forward_samples_total"]
            if compute["student_forward_samples_total"]>0 else np.nan
        )

        self._snapshot("train_exit",step=executed_steps,detailed_memory=True)

        if torch.cuda.is_available():
            self.logs["model"]["train_peak_gpu_allocated_bytes"]=int(torch.cuda.max_memory_allocated(self.device))
            self.logs["model"]["train_peak_gpu_reserved_bytes"]=int(torch.cuda.max_memory_reserved(self.device))
            try:
                self.logs["cuda_allocator_stats_end"]={k:int(v) for k,v in torch.cuda.memory_stats(self.device).items()}
            except Exception:
                self.logs["cuda_allocator_stats_end"]={}

        snaps=self.logs.get("memory_snapshots",[])
        baseline=self.logs.get("memory_baseline",{})

        if snaps:
            def _peak_delta(field):
                vals=[x.get(field) for x in snaps if x.get(field) is not None]
                b=baseline.get(field)
                return max(vals)-b if vals and b is not None else None

            self.logs["model"]["peak_system_ram_delta_gb"]=_peak_delta("system_ram_used_gb")
            self.logs["model"]["peak_process_rss_delta_gb"]=_peak_delta("process_rss_gb")
            self.logs["model"]["peak_process_tree_rss_delta_gb"]=_peak_delta("process_tree_rss_gb")
            self.logs["model"]["peak_process_uss_delta_gb"]=_peak_delta("process_uss_gb")
            self.logs["model"]["peak_process_pss_delta_gb"]=_peak_delta("process_pss_gb")
            self.logs["model"]["peak_process_tree_uss_delta_gb"]=_peak_delta("process_tree_uss_gb")
            self.logs["model"]["peak_process_tree_pss_delta_gb"]=_peak_delta("process_tree_pss_gb")

        self.logs["notes"].append(
            "FLOPs are not profiled inside the timed run. PyTorch profiler with_flops estimates only selected "
            "operators; run FLOP profiling separately if required so benchmark timing is not perturbed."
        )
        self.logs["notes"].append(
            f"Training stopped after optimizer step {executed_steps} while retaining the scheduler/warmup "
            f"configuration for the full requested {self.full_step_budget}-step budget."
        )

        if model_checkpoints is not None: return metrics,model_checkpoints
        return metrics


class PepCL_Retraining(PepCL_Logging_Ext):
    """
    PepCL_Logging_Ext with one optional additional training mode:

        merged_loss=False
            EXACT PepCL_Logging_Ext behavior. train() delegates directly to
            PepCL_Logging_Ext.train().

        merged_loss=True
            Classification-only joint retraining:
              1. sample one MS batch using the existing MS sampler
              2. sample one assay batch using the existing assay sampler
              3. merge the two batches
              4. run ONE student forward pass
              5. compute ONE classification loss over the merged examples
              6. run ONE backward pass / optimizer step

    Nothing about the optimizer, scheduler, warmup, sampling calibration,
    dataset representation, checkpointing, or general logging configuration
    is changed.

    Important
    ---------
    merged_loss=True is intentionally restricted to:
      - classification FT/update data
      - use_replay=True
      - dr_lambda=0
      - identical MS and assay classification loss definitions
      - identical MS/FT loss weights

    This is because a single merged objective cannot preserve two distinct
    loss functions or two distinct scalar objective weights.

    For LogSmoothAP, the merged objective is genuinely:

        LogSmoothAP(concat(MS predictions, FT predictions),
                    concat(MS labels, FT labels))

    It is NOT:
        LogSmoothAP(MS) + LogSmoothAP(FT)

    Thus this implements true joint classification training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        tokenizer,
        device: str,
        mhc_seq_dict: dict,

        steps: int = 200,
        output_dir: str = None,
        save_every_n_steps=None,
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
        allele_aware_sampling: bool = True,

        encoder_lr: float = 2e-4,
        decoder_lr: float = 4e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "linear",
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
        dr_loss_type: str = "mse",

        compact_dataframe: bool = True,
        compact_dtypes: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,

        enable_logging: bool = True,
        telemetry_every_n_steps: int = 10,
        enable_nvml_sampling: bool = True,
        measure_serialized_model_size: bool = False,

        # --------------------------------------------------------------
        # ONLY NEW TRAINING OPTION
        # --------------------------------------------------------------
        merged_loss: bool = False,
    ):
        # Save these only so merged-mode validation can verify that the two
        # classification objectives are genuinely identical.
        self.merged_loss = bool(merged_loss)
        self._retraining_ms_loss_alpha = float(ms_loss_alpha)
        self._retraining_nonms_loss_alpha = float(nonms_loss_alpha)

        # EXACT existing PepCL_Logging_Ext constructor.
        super().__init__(
            model=model,
            train_df=train_df,
            tokenizer=tokenizer,
            device=device,
            mhc_seq_dict=mhc_seq_dict,

            steps=steps,
            output_dir=output_dir,
            save_every_n_steps=save_every_n_steps,
            seed=seed,
            num_workers=num_workers,
            collate_fn=collate_fn,

            num_ms_pos=num_ms_pos,
            ms_neg_ratio=ms_neg_ratio,
            num_nonms_pos=num_nonms_pos,
            nonms_neg_ratio=nonms_neg_ratio,
            ms_batch_size=ms_batch_size,
            nonms_batch_size=nonms_batch_size,

            auto_calibrate=auto_calibrate,
            calibration_steps=calibration_steps,

            continuous_target_col=continuous_target_col,
            continuous_ft_sampling_mode=continuous_ft_sampling_mode,
            pooled_ft_num_pos=pooled_ft_num_pos,
            allele_aware_sampling=allele_aware_sampling,

            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            update_domain_embeddings=update_domain_embeddings,

            use_replay=use_replay,
            primary_task_id=primary_task_id,
            sequential_backprop=sequential_backprop,

            ms_loss_type=ms_loss_type,
            ms_loss_alpha=ms_loss_alpha,
            ms_lambda=ms_lambda,
            nonms_loss_type=nonms_loss_type,
            nonms_loss_alpha=nonms_loss_alpha,
            ft_lambda=ft_lambda,
            dr_lambda=dr_lambda,
            dr_loss_type=dr_loss_type,

            compact_dataframe=compact_dataframe,
            compact_dtypes=compact_dtypes,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,

            enable_logging=enable_logging,
            telemetry_every_n_steps=telemetry_every_n_steps,
            enable_nvml_sampling=enable_nvml_sampling,
            measure_serialized_model_size=measure_serialized_model_size,
        )

        # --------------------------------------------------------------
        # Validate ONLY the new merged-loss mode.
        # Nothing changes when merged_loss=False.
        # --------------------------------------------------------------
        if self.merged_loss:
            if self.ft_is_continuous:
                raise ValueError(
                    "merged_loss=True is classification-only. "
                    "Continuous FT/retraining must use merged_loss=False."
                )

            if not self.use_replay:
                raise ValueError(
                    "merged_loss=True requires use_replay=True because both "
                    "domain 0 (MS) and domain 1 (assay) must be active."
                )

            if self.dr_lambda != 0:
                raise ValueError(
                    "merged_loss=True requires dr_lambda=0. A merged retraining "
                    "objective has no frozen teacher / drift-regularization term."
                )

            if self.ms_loss_type != self.nonms_loss_type:
                raise ValueError(
                    "merged_loss=True requires identical MS and assay loss types. "
                    f"Got ms_loss_type='{self.ms_loss_type}' and "
                    f"nonms_loss_type='{self.nonms_loss_type}'."
                )

            if not np.isclose(
                self._retraining_ms_loss_alpha,
                self._retraining_nonms_loss_alpha
            ):
                raise ValueError(
                    "merged_loss=True requires identical MS and assay loss "
                    "hyperparameters. Got "
                    f"ms_loss_alpha={self._retraining_ms_loss_alpha} and "
                    f"nonms_loss_alpha={self._retraining_nonms_loss_alpha}."
                )

            if not np.isclose(self.ms_lambda, self.ft_lambda):
                raise ValueError(
                    "merged_loss=True requires ms_lambda == ft_lambda because "
                    "there is only one joint loss and therefore no separate "
                    "domain-specific loss weighting."
                )

        self.logs["config"]["merged_loss"] = self.merged_loss

        if self.merged_loss:
            self.logs["notes"].append(
                "Merged-loss retraining is active: one MS batch and one assay "
                "batch are concatenated and optimized using one classification "
                "loss and one student forward/backward pass."
            )

    # ==================================================================
    # Batch merging
    # ==================================================================

    def _merge_retraining_batches(self, batch_ms, batch_ft):
        """
        Merge two already-collated model batches along the batch dimension.

        Handles the only potentially mismatched model dimension explicitly:
        mhc_list / mhc_mask_list may have different max_num_alleles because
        the two batches were collated independently.

        Everything else must have matching non-batch dimensions.
        """

        keys_ms=set(batch_ms)
        keys_ft=set(batch_ft)

        if keys_ms != keys_ft:
            only_ms=sorted(keys_ms-keys_ft)
            only_ft=sorted(keys_ft-keys_ms)
            raise ValueError(
                "MS and assay batches contain different keys and cannot be "
                f"merged safely. MS-only={only_ms}, FT-only={only_ft}"
            )

        def _pad_mhc_axis(x, target_n_alleles, pad_value):
            if x.ndim < 3:
                raise ValueError(
                    "Expected mhc_list/mhc_mask_list to have shape "
                    "[batch, n_alleles, mhc_length]."
                )

            n_missing=target_n_alleles-x.shape[1]

            if n_missing<=0:
                return x

            pad_shape=list(x.shape)
            pad_shape[1]=n_missing

            pad=torch.full(
                pad_shape,
                pad_value,
                dtype=x.dtype,
                device=x.device,
            )

            return torch.cat([x,pad],dim=1)

        merged={}

        for key in batch_ms:
            a=batch_ms[key]
            b=batch_ft[key]

            # ----------------------------------------------------------
            # Tensor fields
            # ----------------------------------------------------------
            if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):

                if key in {"mhc_list","mhc_mask_list"}:
                    target_n_alleles=max(a.shape[1],b.shape[1])

                    if key=="mhc_list":
                        pad_value=int(self.tokenizer.pad_idx)
                    else:
                        pad_value=False

                    a=_pad_mhc_axis(a,target_n_alleles,pad_value)
                    b=_pad_mhc_axis(b,target_n_alleles,pad_value)

                if a.shape[1:] != b.shape[1:]:
                    raise ValueError(
                        f"Cannot merge batch field '{key}': "
                        f"MS shape={tuple(a.shape)}, FT shape={tuple(b.shape)}."
                    )

                merged[key]=torch.cat([a,b],dim=0)

            # ----------------------------------------------------------
            # List / tuple metadata, e.g. allele names
            # ----------------------------------------------------------
            elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
                merged[key]=list(a)+list(b)

            # ----------------------------------------------------------
            # Anything else is unsafe to silently combine.
            # ----------------------------------------------------------
            else:
                raise TypeError(
                    f"Unsupported batch field '{key}' for merged retraining: "
                    f"{type(a).__name__} / {type(b).__name__}"
                )

        return merged

    # ==================================================================
    # Training
    # ==================================================================

    def train(self):
        # --------------------------------------------------------------
        # CRITICAL REPRODUCIBILITY GUARANTEE:
        # merged_loss=False is literally the parent implementation.
        # --------------------------------------------------------------
        if not self.merged_loss:
            return super().train()

        return self._train_merged_classification()

    def _train_merged_classification(self):
        """
        Classification-only joint retraining with one merged objective.

        Scheduler construction is intentionally copied exactly from
        PepCL_Logging_Ext:
            warmup_steps = int(self.steps * self.warmup_ratio)
            scheduler horizon = self.steps

        No scheduler behavior is changed.
        """

        train_call_t0=time.perf_counter()
        self._snapshot("train_enter",detailed_memory=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # ==============================================================
        # Sampling setup: SAME machinery as PepCL_Logging_Ext
        # ==============================================================

        t=time.perf_counter()

        if not self.allele_aware_sampling:
            print(
                "[PepCL_Retraining] Naive balanced sampling: allele-aware "
                "sampling disabled; each stream samples 1:1 positives/negatives "
                "globally. num_*_pos and *_neg_ratio are ignored.\n"
            )

        elif self.auto_calibrate:
            self._auto_calibrate()

        else:
            print(
                f"[PepCL_Retraining] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, "
                f"MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, "
                f"FT neg_ratio={self.ft_cfg['ratio']}\n"
            )

        self.logs["timing"]["sampling_setup_seconds"]=time.perf_counter()-t
        self._snapshot("after_sampling_setup",detailed_memory=True)

        # ==============================================================
        # SAME two samplers / loaders.
        #
        # Difference from standard PepCL:
        # batches are merged BEFORE model forward.
        # ==============================================================

        t=time.perf_counter()
        ms_loader=self._build_loader(0,self.ms_cfg,self.seed)
        self.logs["timing"]["ms_loader_construct_seconds"]=time.perf_counter()-t
        self.logs["data_loader"]["ms"]=self._loader_stats(ms_loader)

        t_iter=time.perf_counter()
        ms_iter=iter(ms_loader)
        self.logs["timing"]["ms_loader_iterator_start_seconds"]=(
            time.perf_counter()-t_iter
        )

        t=time.perf_counter()
        ft_loader=self._build_loader(1,self.ft_cfg,self.seed+1)
        self.logs["timing"]["ft_loader_construct_seconds"]=time.perf_counter()-t
        self.logs["data_loader"]["ft"]=self._loader_stats(ft_loader)

        t_iter=time.perf_counter()
        ft_iter=iter(ft_loader)
        self.logs["timing"]["ft_loader_iterator_start_seconds"]=(
            time.perf_counter()-t_iter
        )

        self._snapshot(
            "after_loader_iterator_start",
            detailed_memory=True,
        )

        # ==============================================================
        # Optimizer / scheduler / scaler
        #
        # EXACT PepCL_Logging_Ext scheduler behavior. DO NOT CHANGE.
        # ==============================================================

        t=time.perf_counter()

        optimizer=self._get_optimizer()

        warmup_steps=int(self.steps*self.warmup_ratio)

        if self.lr_scheduler_type=="linear":
            scheduler=get_linear_schedule_with_warmup(
                optimizer,
                warmup_steps,
                self.steps,
            )

        elif self.lr_scheduler_type=="warmup_constant":
            scheduler=get_constant_schedule_with_warmup(
                optimizer,
                warmup_steps,
            )

        else:
            scheduler=get_constant_schedule(optimizer)

        scaler=GradScaler()

        self.logs["timing"]["optimizer_scheduler_scaler_setup_seconds"]=(
            time.perf_counter()-t
        )

        self._snapshot(
            "after_optimizer_setup_before_state_allocation",
            detailed_memory=True,
        )

        # ==============================================================
        # Training bookkeeping
        # ==============================================================

        metrics=collections.defaultdict(list)

        model_checkpoints=(
            {}
            if self.save_every_n_steps is not None and self.output_dir is None
            else None
        )

        pbar=tqdm(
            range(self.steps),
            desc="PepCL_Retraining",
        )

        compute=self.logs["compute"]
        train_loop_t0=time.perf_counter()
        first_optimizer_snapshot_done=False
        step_wall_times=[]

        # One common criterion. Constructor validation guarantees that the
        # MS and assay criterion definitions are identical.
        merged_criterion=self.criterion_ms

        # Equal/common scalar weight. Normally 1.0.
        merged_lambda=self.ms_lambda

        # ==============================================================
        # Training loop
        # ==============================================================

        for step in pbar:
            step_t0=time.perf_counter()

            self.model.train()
            optimizer.zero_grad()

            # ----------------------------------------------------------
            # Fetch MS batch
            # ----------------------------------------------------------

            t_fetch=time.perf_counter()
            batch_ms=next(ms_iter)
            compute["ms_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

            t_move=time.perf_counter()
            batch_ms=self._move_batch(batch_ms)
            compute["ms_host_to_device_seconds"]+=time.perf_counter()-t_move

            lbl_ms=batch_ms.pop("label")
            ms_n=int(lbl_ms.numel())

            compute["ms_batches"]+=1
            compute["ms_samples"]+=ms_n

            # ----------------------------------------------------------
            # Fetch assay batch
            # ----------------------------------------------------------

            t_fetch=time.perf_counter()
            batch_ft=next(ft_iter)
            compute["ft_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

            t_move=time.perf_counter()
            batch_ft=self._move_batch(batch_ft)
            compute["ft_host_to_device_seconds"]+=time.perf_counter()-t_move

            lbl_ft=batch_ft.pop("label")
            ft_n=int(lbl_ft.numel())

            compute["ft_batches"]+=1
            compute["ft_samples"]+=ft_n

            # ----------------------------------------------------------
            # Merge
            # ----------------------------------------------------------

            t_merge=time.perf_counter()

            merged_batch=self._merge_retraining_batches(
                batch_ms,
                batch_ft,
            )

            merged_labels=torch.cat(
                [lbl_ms,lbl_ft],
                dim=0,
            )

            compute["merged_batch_build_seconds"]+=(
                time.perf_counter()-t_merge
            )

            merged_n=int(merged_labels.numel())

            # ----------------------------------------------------------
            # ONE forward pass + ONE joint loss
            # ----------------------------------------------------------

            with autocast(self.device.type):
                logits=self.model(merged_batch).view(-1)

                merged_loss_raw=merged_criterion(
                    logits,
                    merged_labels,
                )

                merged_loss_weighted=(
                    merged_loss_raw*merged_lambda
                )

            # ----------------------------------------------------------
            # ONE backward pass
            # ----------------------------------------------------------

            scaler.scale(merged_loss_weighted).backward()

            compute["backward_passes"]+=1

            # ----------------------------------------------------------
            # Optimizer / scheduler
            # ----------------------------------------------------------

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            compute["optimizer_steps"]+=1
            compute["scheduler_steps"]+=1

            # Actual forward accounting.
            compute["student_forward_passes_merged"]+=1
            compute["student_forward_samples_merged"]+=merged_n

            # Keep stream-level sample accounting available even though
            # they participated in the same physical forward call.
            compute["student_forward_samples_ms"]+=ms_n
            compute["student_forward_samples_ft"]+=ft_n

            if not first_optimizer_snapshot_done:
                self._snapshot(
                    "after_first_optimizer_step_state_allocated",
                    step=1,
                    detailed_memory=True,
                )
                first_optimizer_snapshot_done=True

            # ----------------------------------------------------------
            # Loss logging
            # ----------------------------------------------------------

            merged_raw_v=float(
                merged_loss_raw.detach().item()
            )

            merged_weighted_v=float(
                merged_loss_weighted.detach().item()
            )

            pbar.set_postfix({
                "Merged":f"{merged_weighted_v:.6f}",
                "MS_n":ms_n,
                "FT_n":ft_n,
            })

            # New explicit retraining trajectories.
            metrics["merged_loss_raw"].append(merged_raw_v)
            metrics["merged_loss_weighted"].append(merged_weighted_v)
            metrics["total_loss"].append(merged_weighted_v)

            # There are deliberately NO separate optimization losses in
            # merged mode. NaN prevents falsely implying that separate MS
            # and FT objectives were optimized.
            metrics["ms_loss"].append(np.nan)
            metrics["ft_loss"].append(np.nan)
            metrics["dr_loss"].append(0.0)

            metrics["ms_classification_loss_raw"].append(np.nan)
            metrics["ms_classification_loss_weighted"].append(np.nan)

            metrics["ft_loss_raw"].append(np.nan)
            metrics["ft_loss_weighted"].append(np.nan)

            metrics["drift_loss_raw"].append(np.nan)
            metrics["drift_loss_weighted"].append(np.nan)

            metrics["continuous_loss_raw"].append(np.nan)
            metrics["continuous_loss_weighted"].append(np.nan)

            # Useful composition diagnostics.
            metrics["merged_ms_n"].append(ms_n)
            metrics["merged_ft_n"].append(ft_n)
            metrics["merged_total_n"].append(merged_n)
            metrics["merged_ms_fraction"].append(
                ms_n/merged_n if merged_n else np.nan
            )

            # ----------------------------------------------------------
            # Checkpoint handling: SAME behavior as parent class
            # ----------------------------------------------------------

            completed_step=step+1

            if self.save_every_n_steps is None:
                should_save=False

            elif isinstance(self.save_every_n_steps,int):
                should_save=(
                    completed_step%self.save_every_n_steps==0
                )

            else:
                should_save=(
                    completed_step in self.save_every_n_steps
                )

            if should_save:
                t_save=time.perf_counter()

                if self.output_dir is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.output_dir,
                            f"pepcl_step_{completed_step}.pt",
                        ),
                    )

                else:
                    model_checkpoints[completed_step]=(
                        copy.deepcopy(self.model).cpu().eval()
                    )

                compute["checkpoint_count"]+=1
                compute["checkpoint_seconds"]+=(
                    time.perf_counter()-t_save
                )

                self._snapshot(
                    f"after_checkpoint_{completed_step}",
                    step=completed_step,
                )

            # ----------------------------------------------------------
            # Telemetry
            # ----------------------------------------------------------

            step_wall_s=time.perf_counter()-step_t0
            step_wall_times.append(step_wall_s)

            compute["sum_step_wall_seconds"]+=step_wall_s

            self._telemetry(
                completed_step,
                step_wall_s,
                completed_step,
                train_loop_t0,
            )

        # ==============================================================
        # End-of-training timing / compute summaries
        # ==============================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        train_loop_seconds=time.perf_counter()-train_loop_t0

        self.logs["timing"]["training_loop_seconds"]=train_loop_seconds
        self.logs["timing"]["train_call_total_seconds"]=(
            time.perf_counter()-train_call_t0
        )
        self.logs["timing"]["end_to_end_seconds"]=(
            time.perf_counter()-self._object_t0
        )

        self.logs["timing"]["average_steps_per_second"]=(
            self.steps/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        self.logs["timing"]["average_seconds_per_step"]=(
            train_loop_seconds/self.steps
            if self.steps>0
            else np.nan
        )

        self.logs["timing"]["first_step_seconds"]=(
            float(step_wall_times[0])
            if step_wall_times
            else np.nan
        )

        if len(step_wall_times)>10:
            steady_s=float(sum(step_wall_times[10:]))
            steady_n=len(step_wall_times)-10

            self.logs["timing"]["steady_state_start_step"]=11

            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=(
                steady_n/steady_s
                if steady_s>0
                else np.nan
            )

            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=(
                steady_s/steady_n
                if steady_n>0
                else np.nan
            )

        else:
            self.logs["timing"]["steady_state_start_step"]=None

            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=np.nan

            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=np.nan

        # --------------------------------------------------------------
        # Compute accounting
        # --------------------------------------------------------------

        compute["student_forward_passes_total"]=(
            compute["student_forward_passes_merged"]
        )

        compute["student_forward_samples_total"]=(
            compute["student_forward_samples_merged"]
        )

        compute["total_forward_passes_including_teacher"]=(
            compute["student_forward_passes_total"]
        )

        compute["total_forward_samples_including_teacher"]=(
            compute["student_forward_samples_total"]
        )

        compute["student_samples_per_second"]=(
            compute["student_forward_samples_total"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["ms_samples_per_second"]=(
            compute["student_forward_samples_ms"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["ft_samples_per_second"]=(
            compute["student_forward_samples_ft"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["teacher_samples_per_second"]=0.0

        compute["seconds_per_1000_student_samples"]=(
            1000.0*train_loop_seconds/
            compute["student_forward_samples_total"]
            if compute["student_forward_samples_total"]>0
            else np.nan
        )

        # ==============================================================
        # Final memory snapshots: SAME general machinery as parent
        # ==============================================================

        self._snapshot(
            "train_exit",
            step=self.steps,
            detailed_memory=True,
        )

        if torch.cuda.is_available():
            self.logs["model"]["train_peak_gpu_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device)
            )

            self.logs["model"]["train_peak_gpu_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device)
            )

            try:
                self.logs["cuda_allocator_stats_end"]={
                    k:int(v)
                    for k,v in torch.cuda.memory_stats(
                        self.device
                    ).items()
                }

            except Exception:
                self.logs["cuda_allocator_stats_end"]={}

        snaps=self.logs.get("memory_snapshots",[])
        baseline=self.logs.get("memory_baseline",{})

        if snaps:
            def _peak_delta(field):
                vals=[
                    x.get(field)
                    for x in snaps
                    if x.get(field) is not None
                ]

                b=baseline.get(field)

                return (
                    max(vals)-b
                    if vals and b is not None
                    else None
                )

            self.logs["model"]["peak_system_ram_delta_gb"]=(
                _peak_delta("system_ram_used_gb")
            )

            self.logs["model"]["peak_process_rss_delta_gb"]=(
                _peak_delta("process_rss_gb")
            )

            self.logs["model"]["peak_process_tree_rss_delta_gb"]=(
                _peak_delta("process_tree_rss_gb")
            )

            self.logs["model"]["peak_process_uss_delta_gb"]=(
                _peak_delta("process_uss_gb")
            )

            self.logs["model"]["peak_process_pss_delta_gb"]=(
                _peak_delta("process_pss_gb")
            )

            self.logs["model"]["peak_process_tree_uss_delta_gb"]=(
                _peak_delta("process_tree_uss_gb")
            )

            self.logs["model"]["peak_process_tree_pss_delta_gb"]=(
                _peak_delta("process_tree_pss_gb")
            )

        self.logs["notes"].append(
            "Merged-loss mode uses one student forward pass and one backward "
            "pass per optimizer step. MS and assay examples are sampled by "
            "their original independent samplers and concatenated before the "
            "forward pass."
        )

        self.logs["notes"].append(
            "In merged-loss mode, component-specific MS and FT loss "
            "trajectories are NaN because those objectives are not separately "
            "optimized. merged_loss_raw / merged_loss_weighted are the actual "
            "optimization trajectories."
        )

        if model_checkpoints is not None:
            return metrics,model_checkpoints

        return metrics

class PepCL_Retraining_Early_Stopping(PepCL_Retraining):
    """
    PepCL_Retraining with fixed checkpoint/early-stopping points.

    `steps` remains the FULL scheduler/warmup budget.

    `early_stopping_points` controls:
      - which model checkpoints are saved
      - the final executed optimizer step = max(early_stopping_points)

    Example:
        steps=15720
        early_stopping_points=[2000,4000,8000]

    executes exactly 8000 optimizer steps, saves checkpoints at 2000/4000/8000,
    and keeps the LR scheduler parameterized exactly as a 15720-step run.

    If early_stopping_points contains one value, training stops at that step and
    the live model is exactly that checkpoint.
    """

    def __init__(self,*args,early_stopping_points=None,**kwargs):
        if early_stopping_points is None:
            raise ValueError("early_stopping_points must be a non-empty list of positive integers.")
        if not isinstance(early_stopping_points,list):
            raise TypeError("early_stopping_points must be a list of positive integers.")
        if len(early_stopping_points)==0:
            raise ValueError("early_stopping_points cannot be empty.")
        if not all(isinstance(x,(int,np.integer)) for x in early_stopping_points):
            raise TypeError("All early_stopping_points values must be integers.")
        if any(int(x)<=0 for x in early_stopping_points):
            raise ValueError("All early_stopping_points values must be > 0.")

        points=sorted({int(x) for x in early_stopping_points})

        if kwargs.get("save_every_n_steps",None) is not None:
            raise ValueError(
                "PepCL_Retraining_Early_Stopping uses early_stopping_points for checkpointing; "
                "save_every_n_steps must be None."
            )

        # The early-stopping points ARE the checkpoint schedule.
        kwargs["save_every_n_steps"]=points

        # Everything else, including model initialization state, sampling seeds,
        # losses, optimizer setup, and merged-loss validation, is exactly the
        # existing PepCL_Retraining constructor.
        super().__init__(*args,**kwargs)

        if points[-1]>self.steps:
            raise ValueError(
                f"Maximum early stopping point ({points[-1]}) cannot exceed "
                f"the full step budget ({self.steps})."
            )

        self.full_step_budget=int(self.steps)
        self.early_stopping_points=points
        self.early_stopping_step=int(points[-1])

        self.logs["config"]["full_step_budget"]=self.full_step_budget
        self.logs["config"]["early_stopping_points"]=list(self.early_stopping_points)
        self.logs["config"]["early_stopping_step"]=self.early_stopping_step
        self.logs["config"]["actual_training_steps"]=self.early_stopping_step
        self.logs["notes"].append(
            "Early-stopping points are also the checkpoint schedule. Training executes only through "
            "the largest requested point, while LR scheduler and warmup remain parameterized by the "
            "full requested step budget."
        )

    def _telemetry(self,step,step_wall_s,completed_steps,train_loop_t0):
        if not self.enable_logging:
            return
        terminal_step=self.early_stopping_step
        if step!=1 and step%self.telemetry_every_n_steps!=0 and step!=terminal_step:
            return
        elapsed=time.perf_counter()-train_loop_t0
        row={
            "step":int(step),
            "step_wall_seconds":float(step_wall_s),
            "cumulative_train_seconds":float(elapsed),
            "cumulative_steps_per_second":float(completed_steps/elapsed) if elapsed>0 else np.nan,
        }
        self._snapshot(f"train_step_{step}",step=step,detailed_memory=False)
        if self.logs["memory_snapshots"]:
            row.update({
                k:v for k,v in self.logs["memory_snapshots"][-1].items()
                if k not in {"tag","step"}
            })
        self.logs["step_telemetry"].append(row)

    def train(self):
        # Standard separate-loss retraining reuses the already-existing
        # PepCL_Logging_Ext early-stopping loop exactly.
        if not self.merged_loss:
            return PepCL_Logging_Ext_Early_Stopping.train(self)

        # Merged classification retraining uses the exact PepCL_Retraining
        # merged loop, except execution ends at the largest early-stop point.
        return self._train_merged_classification_early_stopping()

    def _merge_retraining_batches(self,batch_ms,batch_ft):
        # Reuse the exact existing PepCL_Retraining batch-merging implementation.
        return PepCL_Retraining._merge_retraining_batches(self,batch_ms,batch_ft)

    def _train_merged_classification_early_stopping(self):
        """
        Classification-only joint retraining with one merged objective.

        Scheduler construction is intentionally copied exactly from
        PepCL_Logging_Ext:
            warmup_steps = int(self.steps * self.warmup_ratio)
            scheduler horizon = self.steps

        No scheduler behavior is changed.
        """

        train_call_t0=time.perf_counter()
        self._snapshot("train_enter",detailed_memory=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # ==============================================================
        # Sampling setup: SAME machinery as PepCL_Logging_Ext
        # ==============================================================

        t=time.perf_counter()

        if not self.allele_aware_sampling:
            print(
                "[PepCL_Retraining_Early_Stopping] Naive balanced sampling: allele-aware "
                "sampling disabled; each stream samples 1:1 positives/negatives "
                "globally. num_*_pos and *_neg_ratio are ignored.\n"
            )

        elif self.auto_calibrate:
            self._auto_calibrate()

        else:
            print(
                f"[PepCL_Retraining_Early_Stopping] Manual mode: "
                f"MS n_pos={self.ms_cfg['n_pos']}, "
                f"MS neg_ratio={self.ms_cfg['ratio']} | "
                f"FT n_pos={self.ft_cfg['n_pos']}, "
                f"FT neg_ratio={self.ft_cfg['ratio']}\n"
            )

        self.logs["timing"]["sampling_setup_seconds"]=time.perf_counter()-t
        self._snapshot("after_sampling_setup",detailed_memory=True)

        # ==============================================================
        # SAME two samplers / loaders.
        #
        # Difference from standard PepCL:
        # batches are merged BEFORE model forward.
        # ==============================================================

        t=time.perf_counter()
        ms_loader=self._build_loader(0,self.ms_cfg,self.seed)
        self.logs["timing"]["ms_loader_construct_seconds"]=time.perf_counter()-t
        self.logs["data_loader"]["ms"]=self._loader_stats(ms_loader)

        t_iter=time.perf_counter()
        ms_iter=iter(ms_loader)
        self.logs["timing"]["ms_loader_iterator_start_seconds"]=(
            time.perf_counter()-t_iter
        )

        t=time.perf_counter()
        ft_loader=self._build_loader(1,self.ft_cfg,self.seed+1)
        self.logs["timing"]["ft_loader_construct_seconds"]=time.perf_counter()-t
        self.logs["data_loader"]["ft"]=self._loader_stats(ft_loader)

        t_iter=time.perf_counter()
        ft_iter=iter(ft_loader)
        self.logs["timing"]["ft_loader_iterator_start_seconds"]=(
            time.perf_counter()-t_iter
        )

        self._snapshot(
            "after_loader_iterator_start",
            detailed_memory=True,
        )

        # ==============================================================
        # Optimizer / scheduler / scaler
        #
        # EXACT PepCL_Logging_Ext scheduler behavior. DO NOT CHANGE.
        # ==============================================================

        t=time.perf_counter()

        optimizer=self._get_optimizer()

        warmup_steps=int(self.steps*self.warmup_ratio)

        if self.lr_scheduler_type=="linear":
            scheduler=get_linear_schedule_with_warmup(
                optimizer,
                warmup_steps,
                self.steps,
            )

        elif self.lr_scheduler_type=="warmup_constant":
            scheduler=get_constant_schedule_with_warmup(
                optimizer,
                warmup_steps,
            )

        else:
            scheduler=get_constant_schedule(optimizer)

        scaler=GradScaler()

        self.logs["timing"]["optimizer_scheduler_scaler_setup_seconds"]=(
            time.perf_counter()-t
        )

        self._snapshot(
            "after_optimizer_setup_before_state_allocation",
            detailed_memory=True,
        )

        # ==============================================================
        # Training bookkeeping
        # ==============================================================

        metrics=collections.defaultdict(list)

        model_checkpoints=(
            {}
            if self.save_every_n_steps is not None and self.output_dir is None
            else None
        )

        pbar=tqdm(
            range(self.early_stopping_step),
            desc="PepCL_Retraining_Early_Stopping",
        )

        compute=self.logs["compute"]
        train_loop_t0=time.perf_counter()
        first_optimizer_snapshot_done=False
        step_wall_times=[]

        # One common criterion. Constructor validation guarantees that the
        # MS and assay criterion definitions are identical.
        merged_criterion=self.criterion_ms

        # Equal/common scalar weight. Normally 1.0.
        merged_lambda=self.ms_lambda

        # ==============================================================
        # Training loop
        # ==============================================================

        for step in pbar:
            step_t0=time.perf_counter()

            self.model.train()
            optimizer.zero_grad()

            # ----------------------------------------------------------
            # Fetch MS batch
            # ----------------------------------------------------------

            t_fetch=time.perf_counter()
            batch_ms=next(ms_iter)
            compute["ms_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

            t_move=time.perf_counter()
            batch_ms=self._move_batch(batch_ms)
            compute["ms_host_to_device_seconds"]+=time.perf_counter()-t_move

            lbl_ms=batch_ms.pop("label")
            ms_n=int(lbl_ms.numel())

            compute["ms_batches"]+=1
            compute["ms_samples"]+=ms_n

            # ----------------------------------------------------------
            # Fetch assay batch
            # ----------------------------------------------------------

            t_fetch=time.perf_counter()
            batch_ft=next(ft_iter)
            compute["ft_batch_fetch_seconds"]+=time.perf_counter()-t_fetch

            t_move=time.perf_counter()
            batch_ft=self._move_batch(batch_ft)
            compute["ft_host_to_device_seconds"]+=time.perf_counter()-t_move

            lbl_ft=batch_ft.pop("label")
            ft_n=int(lbl_ft.numel())

            compute["ft_batches"]+=1
            compute["ft_samples"]+=ft_n

            # ----------------------------------------------------------
            # Merge
            # ----------------------------------------------------------

            t_merge=time.perf_counter()

            merged_batch=self._merge_retraining_batches(
                batch_ms,
                batch_ft,
            )

            merged_labels=torch.cat(
                [lbl_ms,lbl_ft],
                dim=0,
            )

            compute["merged_batch_build_seconds"]+=(
                time.perf_counter()-t_merge
            )

            merged_n=int(merged_labels.numel())

            # ----------------------------------------------------------
            # ONE forward pass + ONE joint loss
            # ----------------------------------------------------------

            with autocast(self.device.type):
                logits=self.model(merged_batch).view(-1)

                merged_loss_raw=merged_criterion(
                    logits,
                    merged_labels,
                )

                merged_loss_weighted=(
                    merged_loss_raw*merged_lambda
                )

            # ----------------------------------------------------------
            # ONE backward pass
            # ----------------------------------------------------------

            scaler.scale(merged_loss_weighted).backward()

            compute["backward_passes"]+=1

            # ----------------------------------------------------------
            # Optimizer / scheduler
            # ----------------------------------------------------------

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            compute["optimizer_steps"]+=1
            compute["scheduler_steps"]+=1

            # Actual forward accounting.
            compute["student_forward_passes_merged"]+=1
            compute["student_forward_samples_merged"]+=merged_n

            # Keep stream-level sample accounting available even though
            # they participated in the same physical forward call.
            compute["student_forward_samples_ms"]+=ms_n
            compute["student_forward_samples_ft"]+=ft_n

            if not first_optimizer_snapshot_done:
                self._snapshot(
                    "after_first_optimizer_step_state_allocated",
                    step=1,
                    detailed_memory=True,
                )
                first_optimizer_snapshot_done=True

            # ----------------------------------------------------------
            # Loss logging
            # ----------------------------------------------------------

            merged_raw_v=float(
                merged_loss_raw.detach().item()
            )

            merged_weighted_v=float(
                merged_loss_weighted.detach().item()
            )

            pbar.set_postfix({
                "Merged":f"{merged_weighted_v:.6f}",
                "MS_n":ms_n,
                "FT_n":ft_n,
            })

            # New explicit retraining trajectories.
            metrics["merged_loss_raw"].append(merged_raw_v)
            metrics["merged_loss_weighted"].append(merged_weighted_v)
            metrics["total_loss"].append(merged_weighted_v)

            # There are deliberately NO separate optimization losses in
            # merged mode. NaN prevents falsely implying that separate MS
            # and FT objectives were optimized.
            metrics["ms_loss"].append(np.nan)
            metrics["ft_loss"].append(np.nan)
            metrics["dr_loss"].append(0.0)

            metrics["ms_classification_loss_raw"].append(np.nan)
            metrics["ms_classification_loss_weighted"].append(np.nan)

            metrics["ft_loss_raw"].append(np.nan)
            metrics["ft_loss_weighted"].append(np.nan)

            metrics["drift_loss_raw"].append(np.nan)
            metrics["drift_loss_weighted"].append(np.nan)

            metrics["continuous_loss_raw"].append(np.nan)
            metrics["continuous_loss_weighted"].append(np.nan)

            # Useful composition diagnostics.
            metrics["merged_ms_n"].append(ms_n)
            metrics["merged_ft_n"].append(ft_n)
            metrics["merged_total_n"].append(merged_n)
            metrics["merged_ms_fraction"].append(
                ms_n/merged_n if merged_n else np.nan
            )

            # ----------------------------------------------------------
            # Checkpoint handling: SAME behavior as parent class
            # ----------------------------------------------------------

            completed_step=step+1

            if self.save_every_n_steps is None:
                should_save=False

            elif isinstance(self.save_every_n_steps,int):
                should_save=(
                    completed_step%self.save_every_n_steps==0
                )

            else:
                should_save=(
                    completed_step in self.save_every_n_steps
                )

            if should_save:
                t_save=time.perf_counter()

                if self.output_dir is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.output_dir,
                            f"pepcl_step_{completed_step}.pt",
                        ),
                    )

                else:
                    model_checkpoints[completed_step]=(
                        copy.deepcopy(self.model).cpu().eval()
                    )

                compute["checkpoint_count"]+=1
                compute["checkpoint_seconds"]+=(
                    time.perf_counter()-t_save
                )

                self._snapshot(
                    f"after_checkpoint_{completed_step}",
                    step=completed_step,
                )

            # ----------------------------------------------------------
            # Telemetry
            # ----------------------------------------------------------

            step_wall_s=time.perf_counter()-step_t0
            step_wall_times.append(step_wall_s)

            compute["sum_step_wall_seconds"]+=step_wall_s

            self._telemetry(
                completed_step,
                step_wall_s,
                completed_step,
                train_loop_t0,
            )

        # ==============================================================
        # End-of-training timing / compute summaries
        # ==============================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        executed_steps=self.early_stopping_step
        train_loop_seconds=time.perf_counter()-train_loop_t0

        self.logs["timing"]["training_loop_seconds"]=train_loop_seconds
        self.logs["timing"]["train_call_total_seconds"]=(
            time.perf_counter()-train_call_t0
        )
        self.logs["timing"]["end_to_end_seconds"]=(
            time.perf_counter()-self._object_t0
        )

        self.logs["timing"]["average_steps_per_second"]=(
            executed_steps/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        self.logs["timing"]["average_seconds_per_step"]=(
            train_loop_seconds/executed_steps
            if executed_steps>0
            else np.nan
        )

        self.logs["timing"]["first_step_seconds"]=(
            float(step_wall_times[0])
            if step_wall_times
            else np.nan
        )

        self.logs["timing"]["requested_full_step_budget"]=self.full_step_budget
        self.logs["timing"]["executed_training_steps"]=executed_steps
        self.logs["timing"]["stopped_early"]=bool(executed_steps<self.full_step_budget)

        if len(step_wall_times)>10:
            steady_s=float(sum(step_wall_times[10:]))
            steady_n=len(step_wall_times)-10

            self.logs["timing"]["steady_state_start_step"]=11

            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=(
                steady_n/steady_s
                if steady_s>0
                else np.nan
            )

            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=(
                steady_s/steady_n
                if steady_n>0
                else np.nan
            )

        else:
            self.logs["timing"]["steady_state_start_step"]=None

            self.logs["timing"][
                "steady_state_steps_per_second_after_step_10"
            ]=np.nan

            self.logs["timing"][
                "steady_state_seconds_per_step_after_step_10"
            ]=np.nan

        # --------------------------------------------------------------
        # Compute accounting
        # --------------------------------------------------------------

        compute["student_forward_passes_total"]=(
            compute["student_forward_passes_merged"]
        )

        compute["student_forward_samples_total"]=(
            compute["student_forward_samples_merged"]
        )

        compute["total_forward_passes_including_teacher"]=(
            compute["student_forward_passes_total"]
        )

        compute["total_forward_samples_including_teacher"]=(
            compute["student_forward_samples_total"]
        )

        compute["student_samples_per_second"]=(
            compute["student_forward_samples_total"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["ms_samples_per_second"]=(
            compute["student_forward_samples_ms"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["ft_samples_per_second"]=(
            compute["student_forward_samples_ft"]/train_loop_seconds
            if train_loop_seconds>0
            else np.nan
        )

        compute["teacher_samples_per_second"]=0.0

        compute["seconds_per_1000_student_samples"]=(
            1000.0*train_loop_seconds/
            compute["student_forward_samples_total"]
            if compute["student_forward_samples_total"]>0
            else np.nan
        )

        # ==============================================================
        # Final memory snapshots: SAME general machinery as parent
        # ==============================================================

        self._snapshot(
            "train_exit",
            step=executed_steps,
            detailed_memory=True,
        )

        if torch.cuda.is_available():
            self.logs["model"]["train_peak_gpu_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device)
            )

            self.logs["model"]["train_peak_gpu_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device)
            )

            try:
                self.logs["cuda_allocator_stats_end"]={
                    k:int(v)
                    for k,v in torch.cuda.memory_stats(
                        self.device
                    ).items()
                }

            except Exception:
                self.logs["cuda_allocator_stats_end"]={}

        snaps=self.logs.get("memory_snapshots",[])
        baseline=self.logs.get("memory_baseline",{})

        if snaps:
            def _peak_delta(field):
                vals=[
                    x.get(field)
                    for x in snaps
                    if x.get(field) is not None
                ]

                b=baseline.get(field)

                return (
                    max(vals)-b
                    if vals and b is not None
                    else None
                )

            self.logs["model"]["peak_system_ram_delta_gb"]=(
                _peak_delta("system_ram_used_gb")
            )

            self.logs["model"]["peak_process_rss_delta_gb"]=(
                _peak_delta("process_rss_gb")
            )

            self.logs["model"]["peak_process_tree_rss_delta_gb"]=(
                _peak_delta("process_tree_rss_gb")
            )

            self.logs["model"]["peak_process_uss_delta_gb"]=(
                _peak_delta("process_uss_gb")
            )

            self.logs["model"]["peak_process_pss_delta_gb"]=(
                _peak_delta("process_pss_gb")
            )

            self.logs["model"]["peak_process_tree_uss_delta_gb"]=(
                _peak_delta("process_tree_uss_gb")
            )

            self.logs["model"]["peak_process_tree_pss_delta_gb"]=(
                _peak_delta("process_tree_pss_gb")
            )

        self.logs["notes"].append(
            "Merged-loss mode uses one student forward pass and one backward "
            "pass per optimizer step. MS and assay examples are sampled by "
            "their original independent samplers and concatenated before the "
            "forward pass."
        )

        self.logs["notes"].append(
            "In merged-loss mode, component-specific MS and FT loss "
            "trajectories are NaN because those objectives are not separately "
            "optimized. merged_loss_raw / merged_loss_weighted are the actual "
            "optimization trajectories."
        )

        if model_checkpoints is not None:
            return metrics,model_checkpoints

        return metrics
