"""Per-class confidence distributions for SOTA HydroRecurrentStacker on test.

Reports per true-class:
  - mean/median/std of the predicted-class softmax prob (confidence)
  - mean/median/std of the *correct-class* prob
  - decile bins of confidence
  - confidence | correct vs incorrect
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

_R = Path(__file__).resolve().parents[1]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))

from campaign.eval_recurrent_stacker import scan_split, predict_split, CLASSES
from campaign.train_recurrent_stacker import EnsembleCache
from models.hydro_recurrent_stacker import HydroRecurrentStacker


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    ap.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    ap.add_argument("--ckpt", default="lightning_logs/hydro_recurrent_stacker_combined/best.pt")
    ap.add_argument("--K_eval", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    scales = ck.get("scales", [1, 3, 10])
    model = HydroRecurrentStacker(num_classes=cache.C, ens_n_ckpts=cache.M,
                                  ens_n_classes=cache.C, scales=scales,
                                  gambler=True).to(device)
    model.load_state_dict(ck["state_dict"])

    yt, fp_t, fa_t, stt, _ = predict_split(model, scan_split(args.data_dir, "Test"),
                                           ThreadPoolExecutor(max_workers=12),
                                           device, cache, args.K_eval)
    K = fp_t.shape[1]
    pred = fp_t.argmax(1)
    msp = fp_t.max(1)            # max-softmax-prob (confidence)
    p_true = fp_t[np.arange(len(yt)), yt]  # prob of correct class

    print(f"\nN test = {len(yt)}  classes = {CLASSES}\n")
    print("Per true-class confidence (max-softmax) and prob-of-correct:")
    print(f"{'class':>10}  {'n':>4}  {'conf μ':>7} {'conf σ':>7} {'p_true μ':>9} {'p_true σ':>9}  acc")
    for c in range(K):
        m = (yt == c); n = int(m.sum())
        if n == 0: continue
        cf = msp[m]; pt = p_true[m]
        acc = float((pred[m] == c).mean())
        print(f"{CLASSES[c]:>10}  {n:>4}  {cf.mean():>7.3f} {cf.std():>7.3f}  "
              f"{pt.mean():>9.3f} {pt.std():>9.3f}  {acc:.3f}")

    print("\nConfidence percentiles per true-class (max-softmax):")
    print(f"{'class':>10}  " + " ".join(f"{p:>5}%" for p in (10, 25, 50, 75, 90)))
    for c in range(K):
        m = (yt == c)
        if m.sum() == 0: continue
        q = np.quantile(msp[m], [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"{CLASSES[c]:>10}  " + " ".join(f"{v:>6.3f}" for v in q))

    print("\nConfidence by correct vs incorrect (overall + per-class):")
    correct = pred == yt
    print(f"  overall  correct(n={int(correct.sum())}): μ={msp[correct].mean():.3f}  "
          f"incorrect(n={int((~correct).sum())}): μ={msp[~correct].mean():.3f}")
    for c in range(K):
        m = (yt == c)
        if m.sum() == 0: continue
        cm = correct & m; im = (~correct) & m
        a = msp[cm].mean() if cm.sum() else float('nan')
        b = msp[im].mean() if im.sum() else float('nan')
        print(f"  {CLASSES[c]:>10}  correct(n={int(cm.sum())}): μ={a:.3f}  "
              f"incorrect(n={int(im.sum())}): μ={b:.3f}")

    # Confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(yt, pred): cm[t, p] += 1
    head = " ".join(f"{c[:5]:>6}" for c in CLASSES)
    print("            " + head)
    for i, c in enumerate(CLASSES):
        print(f"  {c:>10}  " + " ".join(f"{cm[i, j]:>6}" for j in range(K)))

    np.savez(Path(args.ckpt).parent / "confidence_dump.npz",
             y=yt, pred=pred, probs=fp_t, abstain=fa_t, source_type=stt)
    print(f"\nwrote {Path(args.ckpt).parent}/confidence_dump.npz")


if __name__ == "__main__":
    raise SystemExit(main())
