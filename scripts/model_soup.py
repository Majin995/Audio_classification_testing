"""Greedy model soup over HydroHydra checkpoints.

Wortsman et al., "Model Soups", ICML 2022 (arXiv:2203.05482).

Greedy soup: starts from the best-val checkpoint and, in descending order
of individual val score, adds candidates to the running average only if
including them does not decrease val/macro_precision. The final averaged
weights are saved as `soup.ckpt` plus a markdown summary.

Soup compatibility: only checkpoints with **identical model architecture**
(same head type, same enabled streams) can be averaged. The script
verifies this by comparing state_dict key sets and tensor shapes.

Usage:
    python -m scripts.model_soup \
        --ckpts <c1>.ckpt <c2>.ckpt <c3>.ckpt \
        --data_dir $DATA_DIR \
        --out_dir lightning_logs/phaseI_soup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_hydra import HydroHydra


def _check_compatible(state_dicts: list[dict], names: list[str]) -> None:
    ref_keys = set(state_dicts[0].keys())
    ref_shapes = {k: v.shape for k, v in state_dicts[0].items()}
    for name, sd in zip(names[1:], state_dicts[1:]):
        keys = set(sd.keys())
        if keys != ref_keys:
            extra = keys - ref_keys
            missing = ref_keys - keys
            raise SystemExit(
                f"[soup] {name} has incompatible keys vs {names[0]}\n"
                f"  extra: {sorted(extra)[:5]}\n"
                f"  missing: {sorted(missing)[:5]}"
            )
        for k, v in sd.items():
            if v.shape != ref_shapes[k]:
                raise SystemExit(
                    f"[soup] {name} key {k} shape {tuple(v.shape)} != "
                    f"{tuple(ref_shapes[k])} ({names[0]})"
                )


def _average(state_dicts: list[dict]) -> dict:
    n = len(state_dicts)
    out = {}
    for k in state_dicts[0].keys():
        v = state_dicts[0][k]
        if v.dtype.is_floating_point:
            stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
            out[k] = stacked.mean(dim=0).to(v.dtype)
        else:
            # Integer buffers (e.g. num_batches_tracked) — keep first.
            out[k] = v.clone()
    return out


@torch.no_grad()
def _eval(model, loader, device, num_classes: int) -> tuple[float, dict]:
    """Returns (macro_P, per_class_P_dict)."""
    from torchmetrics.classification import MulticlassPrecision
    model.eval()
    macro = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    per = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        if out.size(-1) > num_classes:
            out = out[:, :num_classes]
        macro.update(out, y)
        per.update(out, y)
    return float(macro.compute().item()), {
        f"c{i}": float(v) for i, v in enumerate(per.compute().tolist())
    }


def main():
    ap = argparse.ArgumentParser(description="Greedy model soup for HydroHydra")
    ap.add_argument("--ckpts", nargs="+", required=True, help="Checkpoints to soup")
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=5_120)
    ap.add_argument("--fixed_len", type=int, default=5_120)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if not args.data_dir:
        raise SystemExit("Set --data_dir or export DATA_DIR")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load every ckpt, evaluate val individually ────────────────────
    print(f"[soup] loading {len(args.ckpts)} checkpoints …")
    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size, num_threads=args.num_threads,
        target_sr=args.sample_rate, fixed_len=args.fixed_len,
        oversample_train=False,
    )
    data.setup()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
    nc = data.num_classes

    individuals = []
    state_dicts = []
    for ck in args.ckpts:
        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False)
        m = m.to(device)
        sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        state_dicts.append(sd)
        val_mp, _ = _eval(m, val_loader, device, nc)
        individuals.append((ck, val_mp))
        print(f"  {Path(ck).name}: val/μP = {val_mp:.4f}")
        del m; torch.cuda.empty_cache()

    _check_compatible(state_dicts, [c for c, _ in individuals])

    # ── 2. Greedy soup, descending by individual val score ───────────────
    order = sorted(range(len(individuals)), key=lambda i: -individuals[i][1])
    in_soup = [order[0]]
    soup_sd = state_dicts[order[0]]
    soup_val = individuals[order[0]][1]
    print(f"\n[soup] start with {Path(args.ckpts[order[0]]).name} "
          f"(val={soup_val:.4f})")

    for idx in order[1:]:
        cand_in = in_soup + [idx]
        avg_sd = _average([state_dicts[i] for i in cand_in])
        # Build a fresh model on the first ckpt (any ckpt works since all
        # share architecture) and load the averaged state_dict.
        m = HydroHydra.load_from_checkpoint(args.ckpts[order[0]], map_location=device, strict=False).to(device)
        m.load_state_dict(avg_sd, strict=False)
        cand_val, _ = _eval(m, val_loader, device, nc)
        del m; torch.cuda.empty_cache()
        ckpt_name = Path(args.ckpts[idx]).name
        if cand_val >= soup_val - 1e-5:
            in_soup = cand_in
            soup_sd = avg_sd
            soup_val = cand_val
            print(f"  + {ckpt_name}  → soup val/μP = {cand_val:.4f}  "
                  f"(kept; {len(in_soup)} ckpts)")
        else:
            print(f"  - {ckpt_name}  → would drop val to {cand_val:.4f} "
                  f"(rejected)")

    # ── 3. Final eval on val + test, save artifacts ──────────────────────
    print(f"\n[soup] FINAL: {len(in_soup)}/{len(args.ckpts)} ckpts, "
          f"val/μP = {soup_val:.4f}")
    m = HydroHydra.load_from_checkpoint(args.ckpts[order[0]], map_location=device, strict=False).to(device)
    m.load_state_dict(soup_sd, strict=False)
    final_val, val_per = _eval(m, val_loader, device, nc)
    final_test, test_per = _eval(m, test_loader, device, nc)
    print(f"[soup] val/μP = {final_val:.4f}  test/μP = {final_test:.4f}")

    # Save the soup as a Lightning-style ckpt for inference compatibility.
    soup_path = out_dir / "soup.ckpt"
    # Reload one ckpt to get the hparams payload, then swap state_dict.
    base = torch.load(args.ckpts[order[0]], map_location="cpu", weights_only=False)
    base["state_dict"] = soup_sd
    base["soup_components"] = [Path(args.ckpts[i]).name for i in in_soup]
    torch.save(base, soup_path)
    print(f"[soup] saved → {soup_path}")

    # Markdown summary.
    md = [f"# Model Soup — {len(in_soup)}/{len(args.ckpts)} ckpts kept", ""]
    md.append("## Components")
    for i in in_soup:
        md.append(f"- `{args.ckpts[i]}` (individual val/μP={individuals[i][1]:.4f})")
    md.append("")
    md.append("## Rejected")
    for i in order:
        if i not in in_soup:
            md.append(f"- `{args.ckpts[i]}` (individual val/μP={individuals[i][1]:.4f})")
    md.append("")
    md.append("## Final metrics")
    md.append(f"- val/macro_P = **{final_val:.4f}**")
    md.append(f"- test/macro_P = **{final_test:.4f}**")
    md.append(f"- val/test gap = {final_val - final_test:+.4f}")
    md.append("")
    md.append("### Per-class precision")
    md.append("| split | c0 | c1 | c2 | c3 |")
    md.append("|---|---|---|---|---|")
    md.append(f"| val | {val_per['c0']:.4f} | {val_per['c1']:.4f} | "
              f"{val_per['c2']:.4f} | {val_per['c3']:.4f} |")
    md.append(f"| test | {test_per['c0']:.4f} | {test_per['c1']:.4f} | "
              f"{test_per['c2']:.4f} | {test_per['c3']:.4f} |")

    (out_dir / "soup.md").write_text("\n".join(md) + "\n")
    (out_dir / "soup.json").write_text(json.dumps({
        "components": [args.ckpts[i] for i in in_soup],
        "rejected":   [args.ckpts[i] for i in order if i not in in_soup],
        "val_macro_P":  final_val,
        "test_macro_P": final_test,
        "val_per_class":  val_per,
        "test_per_class": test_per,
    }, indent=2))
    print(f"[soup] saved → {out_dir / 'soup.md'}")


if __name__ == "__main__":
    main()
