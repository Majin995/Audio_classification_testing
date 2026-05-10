"""Dump softmax probs for each checkpoint over a given dataset's val+test
splits to .npz, so downstream calibration sweeps can iterate cheaply
without re-running the model forward.

Outputs: <out_dir>/<ckpt_stem>.npz with {val_probs, val_y, test_probs, test_y}.
Also writes <out_dir>/_meta.json with class names, num_classes, ckpt list.
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
def collect(model, loader, device, num_classes):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if out.size(-1) > num_classes:
            out = out[:, :num_classes]
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
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
    )
    dm.setup()
    nc = dm.num_classes
    classes = [dm.idx_to_class[i] for i in range(nc)]
    print(f"[dump] data_dir={args.data_dir} num_classes={nc} classes={classes}")

    meta = {
        "data_dir": args.data_dir,
        "num_classes": nc,
        "classes": classes,
        "ckpts": [],
    }

    for ck in args.ckpts:
        stem = Path(ck).stem
        out_path = out / f"{stem}.npz"
        if out_path.exists():
            print(f"[dump] skip (exists): {out_path}")
            meta["ckpts"].append({"ckpt": ck, "stem": stem, "npz": str(out_path)})
            continue
        print(f"[dump] {ck}")
        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False)
        m = m.to(device).eval()
        # Re-setup dm before each model since DALI iterators are one-shot
        out_dict = {}
        for split_name, getter, key_p, key_y in [
            ("train", dm.train_dataloader, "train_probs", "train_y"),
            ("val",   dm.val_dataloader,   "val_probs",   "val_y"),
            ("test",  dm.test_dataloader,  "test_probs",  "test_y"),
        ]:
            try:
                dm.setup()
                P_, Y_ = collect(m, getter(), device, nc)
                out_dict[key_p] = P_
                out_dict[key_y] = Y_
            except FileNotFoundError as e:
                print(f"  no {split_name} split: {e}")
            except Exception as e:
                print(f"  {split_name} failed: {e}")
        if not out_dict:
            print(f"  ! nothing dumped, skipping")
            del m
            torch.cuda.empty_cache()
            continue
        np.savez(out_path, **out_dict)
        sizes = "  ".join(f"{k}={v.shape}" for k, v in out_dict.items() if v.ndim > 0)
        print(f"  wrote {out_path}  {sizes}")
        meta["ckpts"].append({"ckpt": ck, "stem": stem, "npz": str(out_path)})
        del m
        torch.cuda.empty_cache()

    with open(out / "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump] done → {out}/_meta.json")


if __name__ == "__main__":
    main()
