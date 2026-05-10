"""HydroHydra student model — distills 4 binary specialist teachers.

Each teacher is a HydroHydra binary classifier {pos_class, neg, abstain}
trained per-class with the gambler's loss. The student is a standard
4-way + abstain HydroHydra trained with a combined loss:

    L_student = α · T² · KL(student || teacher_consensus, T) + (1-α) · CE_gambler

where the per-sample teacher consensus is constructed as:

    votes_c       = teacher_c.softmax(logits)[..., 0]   # P[pos] for class c
    abstains_c    = teacher_c.softmax(logits)[..., 2]   # P[abstain]
    mean_abstain  = mean(abstains_c)                    # 1 number per sample
    votes_norm    = votes / sum(votes)                  # 4-way over classes
    target_5way   = [(1-ab) * votes_norm, ab]           # 5-way distribution

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network"
(arXiv:1503.02531). Multi-teacher distillation: Tan et al., "Multilingual
NMT with Knowledge Distillation" (arXiv:1902.10461).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hydro_hydra import HydroHydra


class HydroHydraStudent(HydroHydra):
    """4-way + abstain student that distills from 4 binary specialist teachers.

    The student is architecturally identical to a vanilla HydroHydra — the
    only difference is ``training_step``, which adds a KL term against the
    teacher consensus.
    """

    def __init__(self, *args, kd_alpha: float = 0.7, kd_temperature: float = 4.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha       = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        self._teachers: List[HydroHydra] = []
        # Save kd-* hparams alongside the rest of the student's hparams.
        self.save_hyperparameters({
            "kd_alpha": self.kd_alpha,
            "kd_temperature": self.kd_temperature,
        })

    def set_teachers(self, teachers: List[HydroHydra]) -> None:
        """Single teacher per class (legacy path). teachers must be in
        alphabetical class order: Cargo, Passenger, Tanker, Tug."""
        for t in teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)
        # Use a plain Python list — not nn.ModuleList — so the teachers'
        # parameters are NOT registered as student parameters (no optimizer
        # touch, no checkpoint bloat).
        self._teachers = teachers
        self._teacher_groups: List[List[HydroHydra]] = [[t] for t in teachers]

    def set_teacher_groups(self, groups: List[List[HydroHydra]]) -> None:
        """Multi-teacher-per-class. ``groups[c]`` is the list of teachers
        for class ``c`` (e.g. v1 + v2 + v4 for Cargo). Per-sample per-class
        votes are averaged within each group before consensus."""
        flat: List[HydroHydra] = []
        for g in groups:
            for t in g:
                t.eval()
                for p in t.parameters():
                    p.requires_grad_(False)
                flat.append(t)
        self._teachers = flat
        self._teacher_groups = groups

    def _teachers_to(self, device):
        for t in self._teachers:
            t.to(device)

    @torch.no_grad()
    def _teacher_consensus(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the 5-way teacher consensus distribution per sample.

        Each teacher emits 3 logits ``[pos, neg, abstain]``. The consensus is

            target[c]   = (1 - mean_abstain) * vote_c / sum_c vote_c   for c=0..3
            target[4]   = mean_abstain

        Returns ``(B, 5)`` — a valid probability distribution per row.
        """
        T = self.kd_temperature
        votes      = []
        abstains   = []
        for group in self._teacher_groups:
            # Per-class group: average votes & abstains across the
            # teachers for that class, then aggregate across classes.
            grp_votes    = []
            grp_abstains = []
            for teacher in group:
                full = teacher(x)                   # (B, 3): [pos, neg, abst]
                probs = F.softmax(full / T, dim=-1)
                grp_votes.append(probs[..., 0])
                grp_abstains.append(probs[..., 2])
            votes.append(torch.stack(grp_votes, dim=-1).mean(dim=-1))
            abstains.append(torch.stack(grp_abstains, dim=-1).mean(dim=-1))
        votes      = torch.stack(votes, dim=-1)     # (B, num_classes)
        abstains   = torch.stack(abstains, dim=-1)  # (B, num_classes)
        mean_ab    = abstains.mean(dim=-1, keepdim=True)         # (B, 1)
        votes_norm = votes / votes.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target     = torch.cat([
            (1.0 - mean_ab) * votes_norm,
            mean_ab,
        ], dim=-1).clamp_min(1e-8)
        # Re-normalize numerically (clamp_min may break unit sum).
        target = target / target.sum(dim=-1, keepdim=True)
        return target

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_aug = self.wave_aug(x)

        # Standard student forward + supervised loss.
        logits  = self(x_aug, y)                    # (B, 5)  full incl abstain
        sup_loss = self._compute_loss(logits, y)
        class_logits = self._class_logits(logits)
        self.train_acc(class_logits, y)

        # ── KD loss ────────────────────────────────────────────────────
        kd_loss = torch.tensor(0.0, device=logits.device)
        if self._teachers and self.kd_alpha > 0.0:
            self._teachers_to(logits.device)
            target = self._teacher_consensus(x)     # (B, 5)
            T = self.kd_temperature
            log_s = F.log_softmax(logits / T, dim=-1)
            kl = F.kl_div(log_s, target, reduction="batchmean")
            kd_loss = (T * T) * kl

        loss = (1.0 - self.kd_alpha) * sup_loss + self.kd_alpha * kd_loss

        self.log("train/loss",     loss,     on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/sup_loss", sup_loss, on_step=False, on_epoch=True)
        self.log("train/kd_loss",  kd_loss,  on_step=False, on_epoch=True)
        self.log("train/acc",      self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
