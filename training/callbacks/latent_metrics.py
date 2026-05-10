"""
LatentMetricsCallback — diagnostics for class-separability of the learned
post-pool embedding (HydroPreciseV2._features output).

Logs four metrics per validation epoch (default every epoch):

  latent/silhouette       : sklearn silhouette score (cosine), ∈ [-1, 1]; >0 means
                            on average a sample is closer to its own class
                            cluster than to the nearest other class.
  latent/fisher_ratio     : tr(S_B) / tr(S_W). Higher = better class separation
                            in the *raw* (non-normalized) feature space.
  latent/knn_acc          : 5-NN purity on a held-out half of val embeddings
                            (cosine distance). A direct test of how well the
                            embedding alone — without the classification head —
                            separates classes.
  latent/centroid_cos_sim : Mean off-diagonal cosine similarity of class
                            centroids. Lower = centroids more orthogonal.
  latent/intra_class_var  : Mean intra-class L2 variance (lower = tighter clusters).

All metrics are computed on the embedding produced by `pl_module._features(x)`
which must return a (B, D) tensor. The callback subsamples to ``sample_size``
rows for the silhouette/k-NN steps (those are O(n²) and O(n·k) respectively).
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch_lightning import Callback
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier


class LatentMetricsCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int  = 1,
        sample_size:    int  = 2000,
        knn_k:          int  = 5,
        seed:           int  = 0,
    ):
        super().__init__()
        self.every_n     = max(1, every_n_epochs)
        self.sample_size = sample_size
        self.knn_k       = knn_k
        self.seed        = seed

    @torch.no_grad()
    def _collect(self, pl_module, val_dataloader) -> tuple[np.ndarray, np.ndarray]:
        was_training = pl_module.training
        pl_module.eval()
        feats, labs = [], []
        for batch in val_dataloader:
            x, y = batch
            x = x.to(pl_module.device, non_blocking=True)
            feats.append(pl_module._features(x).float().cpu())
            labs.append(y.cpu())
        if was_training:
            pl_module.train()
        return torch.cat(feats).numpy(), torch.cat(labs).numpy()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during sanity-check val, and respect the cadence.
        if trainer.sanity_checking:
            return
        if (pl_module.current_epoch + 1) % self.every_n != 0:
            return
        if not hasattr(pl_module, "_features"):
            return

        try:
            val_dl = trainer.datamodule.val_dataloader()
        except Exception:
            return

        feats, labs = self._collect(pl_module, val_dl)
        if len(np.unique(labs)) < 2:
            return

        # Subsample for silhouette / KNN if too big
        n = len(feats)
        rng = np.random.default_rng(self.seed)
        if self.sample_size and n > self.sample_size:
            idx = rng.choice(n, self.sample_size, replace=False)
            sf, sl = feats[idx], labs[idx]
        else:
            sf, sl = feats, labs

        # ── Silhouette (cosine) ─────────────────────────────────────────
        try:
            sil = float(silhouette_score(sf, sl, metric="cosine"))
        except Exception:
            sil = float("nan")

        # ── Fisher trace ratio ──────────────────────────────────────────
        global_mu = feats.mean(axis=0, keepdims=True)
        sw = sb = 0.0
        intra_var_per_class = []
        for c in np.unique(labs):
            mask = labs == c
            cf = feats[mask]
            mu = cf.mean(axis=0, keepdims=True)
            sw += float(((cf - mu) ** 2).sum())
            sb += float(mask.sum() * ((mu - global_mu) ** 2).sum())
            intra_var_per_class.append(float(((cf - mu) ** 2).sum() / max(mask.sum(), 1)))
        fisher = sb / max(sw, 1e-9)
        intra_var = float(np.mean(intra_var_per_class))

        # ── k-NN purity on a held-out half ──────────────────────────────
        try:
            half = len(sf) // 2
            knn = KNeighborsClassifier(n_neighbors=self.knn_k, metric="cosine")
            knn.fit(sf[:half], sl[:half])
            knn_acc = float(knn.score(sf[half:], sl[half:]))
        except Exception:
            knn_acc = float("nan")

        # ── Centroid cosine separation ──────────────────────────────────
        centroids = np.stack(
            [feats[labs == c].mean(axis=0) for c in np.unique(labs)]
        )
        cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        cos = cn @ cn.T
        K = len(centroids)
        if K > 1:
            iu = np.triu_indices(K, k=1)
            mean_inter_cos = float(cos[iu].mean())
        else:
            mean_inter_cos = 0.0

        # ── Log everything via the LightningModule so it lands on every logger ─
        pl_module.log("latent/silhouette",       sil,           prog_bar=False)
        pl_module.log("latent/fisher_ratio",     fisher,        prog_bar=False)
        pl_module.log("latent/knn_acc",          knn_acc,       prog_bar=True)
        pl_module.log("latent/centroid_cos_sim", mean_inter_cos, prog_bar=False)
        pl_module.log("latent/intra_class_var",  intra_var,     prog_bar=False)
