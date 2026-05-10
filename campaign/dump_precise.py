"""Dump probs for HydroPrecise ckpts (separate from Hydra)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_precise_v2 import HydroPreciseV2


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
    ap.add_argument("--batch_size", type=int, default=32)
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
    print(f"[dump-precise] data_dir={args.data_dir} num_classes={nc}")

    meta = {"data_dir": args.data_dir, "num_classes": nc, "classes": classes,
            "ckpts": []}
    for ck in args.ckpts:
        stem = Path(ck).stem
        out_path = out / f"{stem}.npz"
        if out_path.exists():
            print(f"[dump-precise] skip (exists): {out_path}")
            meta["ckpts"].append({"ckpt": ck, "stem": stem, "npz": str(out_path)})
            continue
        print(f"[dump-precise] {ck}")
        try:
            m = HydroPreciseV2.load_from_checkpoint(ck, map_location=device, strict=False)
        except Exception as e:
            print(f"  ! load failed: {e}")
            continue
        m = m.to(device).eval()
        out_dict = {}
        try:
            dm.setup()
            vp, vy = collect(m, dm.val_dataloader(), device, nc)
            out_dict["val_probs"], out_dict["val_y"] = vp, vy
        except FileNotFoundError as e:
            print(f"  no val: {e}")
        except Exception as e:
            print(f"  val failed: {e}")
        try:
            dm.setup()
            tp, ty = collect(m, dm.test_dataloader(), device, nc)
            out_dict["test_probs"], out_dict["test_y"] = tp, ty
        except FileNotFoundError as e:
            print(f"  no test: {e}")
        except Exception as e:
            print(f"  test failed: {e}")
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
