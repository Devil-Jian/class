"""cectcls: prepare, train, select, match, export synthesis, evaluate."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .manifest import prepare
from .audit import audit_images
from .cohort import build_cohort
from .locked_cohort import lock_cohort
from .png_export import export_png
from .matching import match_ncct
from .selection import export_selection
from .synthesis import export_requests, register_outputs
from .train import evaluate_checkpoint, fit


def _training_options(parser, include_stage=True):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    if include_stage:
        parser.add_argument("--stage", choices=["selection", "selected"], default="selected")
    parser.add_argument("--selection")
    parser.add_argument("--syn-index")
    parser.add_argument("--png-index")
    parser.add_argument("--group", default="cect_real",
                        choices=["ncct", "cect_real", "cect_syn", "cect_real_aug", "cect_real_syn"])
    parser.add_argument("--model", default="dinov2s_mil",
                        choices=["resnet18_mil", "deit_mil", "dinov2s_mil"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--encode-chunk", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--uniform", type=int, default=16)
    parser.add_argument("--random-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["sqrt", "none"], default="sqrt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True)


def make_parser():
    parser = argparse.ArgumentParser(prog="cectcls")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lock-cohort")
    p.add_argument("--source", required=True)
    p.add_argument("--image-root", help="Override only the parent directory of the 1298 patient scans")
    p.add_argument("--splits-source", help="Previously fixed splits.csv, preserved verbatim by patient ID")
    p.add_argument("--out", required=True)
    p.add_argument("--expected-count", type=int, default=1298)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--without-splits", action="store_true")

    p = sub.add_parser("prepare-cohort")
    p.add_argument("--metadata", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label-column", choices=["三分类标签", "主类别"], default="三分类标签")
    p.add_argument("--grades", nargs="+", default=["GOLD", "SILVER", "REJECT"])
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-missing-mask", action="store_true")

    p = sub.add_parser("audit-images")
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strict-mask", action="store_true")
    p.add_argument("--workers", type=int, default=2, help="Header QC threads; 1-2 suggested for HDD")

    p = sub.add_parser("prepare")
    p.add_argument("--label-map")
    p.add_argument("--header-cache", help="Validated audit-images image_header_cache.json")
    p.add_argument("--no-mask-cache", action="store_true")
    p.add_argument("--cases", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("train")
    _training_options(p)

    p = sub.add_parser("train-grid")
    _training_options(p)
    p.set_defaults(stage="selected")

    p = sub.add_parser("export-selection")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int)
    p.add_argument("--out", required=True)

    p = sub.add_parser("match-ncct")
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--radius-mm", type=float, default=15.)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--out", required=True)

    p = sub.add_parser("export-syn-requests")
    p.add_argument("--manifest", required=True)
    p.add_argument("--matched", required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--out-root", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("export-png")
    p.add_argument("--manifest", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--syn-index")
    p.add_argument("--out", required=True)

    p = sub.add_parser("register-syn")
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("evaluate")
    _training_options(p, include_stage=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "lock-cohort":
        print(lock_cohort(args.source, args.out, expected_count=args.expected_count,
                          seed=args.seed, generate_splits=not args.without_splits,
                          image_root=args.image_root, splits_source=args.splits_source))
    elif args.command == "prepare-cohort":
        print(build_cohort(args.metadata, args.image_root, args.out,
                           label_column=args.label_column, grades=args.grades,
                           train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                           seed=args.seed, require_mask=not args.allow_missing_mask))
    elif args.command == "audit-images":
        print(audit_images(args.cases, args.out, strict_mask=args.strict_mask,
                           workers=args.workers))
    elif args.command == "prepare":
        records = prepare(args.cases, args.splits, args.out, label_map_path=args.label_map,
                          header_cache_path=args.header_cache,
                          reuse_mask_cache=not args.no_mask_cache)
        print(f"Prepared {len(records)} patients")
    elif args.command == "train":
        print(fit(args))
    elif args.command == "train-grid":
        if not args.selection:
            raise ValueError("--selection required for train-grid")
        for model in ("resnet18_mil", "deit_mil", "dinov2s_mil"):
            for group in ("ncct", "cect_real", "cect_syn", "cect_real_aug", "cect_real_syn"):
                args.model, args.group = model, group
                args.out = str(Path(args.out) / model / group)
                print(f"Training: {model}/{group}", flush=True)
                fit(args)
                # Restore output root rather than nesting subsequent experiments.
                args.out = str(Path(args.out).parents[1])
    elif args.command == "export-selection":
        print(f"Selected {len(export_selection(args.manifest, args.checkpoint, args.out, args.top_k, args.device, args.image_size))} patients")
    elif args.command == "match-ncct":
        print(f"Matched {len(match_ncct(args.manifest, args.selection, args.out, args.radius_mm, args.stride))} patients")
    elif args.command == "export-syn-requests":
        print(f"Exported {len(export_requests(args.matched, args.out_root, args.out, args.manifest, args.split))} requests")
    elif args.command == "export-png":
        print(f"Exported {len(export_png(args.manifest, args.selection, args.out, args.syn_index))} PNGs")
    elif args.command == "register-syn":
        print(f"Registered {len(register_outputs(args.requests, args.out))} synthetic images")
    elif args.command == "evaluate":
        print(evaluate_checkpoint(args))


if __name__ == "__main__":
    main()
