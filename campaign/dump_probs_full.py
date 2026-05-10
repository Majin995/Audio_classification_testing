"""Dump softmax probs over the FULL output vector (including abstain logit).

Hydra outputs 5 logits: 4 class + 1 abstain. The standard dump strips to 4.
This dump preserves all 5 to allow stackers to use the abstain signal as
an uncertainty feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_hydra import HydroHydra


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        # Keep ALL logits (do NOT strip abstain)
        P.append(F.softmax(out.float(), dim=-1).cpu().numpy())
        Y.append(y.detach().cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=5120)
    ap.add_argument("--fixed_len", type=int, default=5120)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DALIAudioDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_threads=args.num_threads, target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
    )
    dm.setup()
    nc = dm.num_classes
    classes = [dm.idx_to_class[i] for i in range(nc)]
    print(f"[dump-full] data_dir={args.data_dir} num_classes={nc} (preserving abstain logit)")
    meta = {"data_dir": args.data_dir, "num_classes": nc,
            "classes": classes, "ckpts": []}

    for ck in args.ckpts:
        stem = Path(ck).stem
        out_path = out / f"{stem}.npz"
        if out_path.exists():
            print(f"[dump-full] skip (exists): {out_path}")
            meta["ckpts"].append({"ckpt": ck, "stem": stem, "npz": str(out_path)})
            continue
        print(f"[dump-full] {ck}")
        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False)
        m = m.to(device).eval()
        out_dict = {}
        for split, getter, kp, ky in [
            ("val", dm.val_dataloader, "val_probs", "val_y"),
            ("test", dm.test_dataloader, "test_probs", "test_y"),
        ]:
            try:
                dm.setup()
                P_, Y_ = collect(m, getter(), device)
                out_dict[kp] = P_
                out_dict[ky] = Y_
            except FileNotFoundError as e:
                print(f"  no {split}: {e}")
        if out_dict:
            np.savez(out_path, **out_dict)
            sizes = "  ".join(f"{k}={v.shape}" for k, v in out_dict.items() if v.ndim > 0)
            print(f"  wrote {out_path}  {sizes}")
            meta["ckpts"].append({"ckpt": ck, "stem": stem, "npz": str(out_path)})
        del m
        torch.cuda.empty_cache()
    with open(out / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
