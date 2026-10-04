#!/usr/bin/env python3
"""Retrain strict known-TRAIN species/parent holdout folds without real unknowns."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosafe_support import pipeline, protocol
from taxosafe_support.holdout import build_folds, run_validation, select_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", choices=protocol.VARIANTS, default="main")
    parser.add_argument("--kind", choices=("species", "parent", "both"), default="both")
    parser.add_argument("--fold-id", action="append", default=[], help="Repeat to select exact eligible fold IDs")
    parser.add_argument("--max-folds", type=int, default=0, help="Seeded subset size; zero runs all eligible folds")
    parser.add_argument("--known-val-fraction", type=float, default=.2)
    parser.add_argument("--min-train-per-leaf", type=int, default=2)
    parser.add_argument("--run-dir", default="runs/taxosafe_new/strict_holdout")
    parser.add_argument("--preflight", action="store_true", help="Print TRAIN-only fold plan without CLIP/CUDA")
    args = parser.parse_args()
    cfg = protocol.effective_config(protocol.resolve(args.config), args.variant, args.seed)
    meta = pipeline.hierarchy(cfg)
    # Deliberately do not call load_stage_rows('train'): that would also read
    # the external val_known manifest. All folds originate in TRAIN only.
    rows = protocol.read_split(cfg, "train", meta)
    eligible = build_folds(rows, meta, seed=args.seed, kind=args.kind,
                           known_val_fraction=args.known_val_fraction,
                           min_train_per_leaf=args.min_train_per_leaf)
    folds = select_folds(eligible, args.seed, args.fold_id, args.max_folds)
    if not folds:
        raise SystemExit("No eligible folds with independent inner known validation")
    if args.preflight:
        print(json.dumps({"source_split": "train", "eligible_fold_count": len(eligible),
                          "selected_fold_count": len(folds),
                          "folds": [{"id": fold["id"], "kind": fold["kind"],
                                     "heldout_leaf_ids": fold["heldout_leaf_ids"],
                                     "gradient_image_count": len(fold["train_indices"]),
                                     "inner_known_validation_count": len(fold["val_known_indices"]),
                                     "heldout_image_count": len(fold["heldout_indices"]),
                                     "inner_validation_missing_leaf_ids": fold["inner_validation_missing_leaf_ids"]}
                                    for fold in folds],
                          "real_unknown_or_test_inputs_used": False}, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("Strict MaPLe fold retraining requires CUDA; --preflight and CPU contract tests are available")
    directory = protocol.resolve(args.run_dir)
    with protocol.run_lock(directory):
        summary = run_validation(cfg, directory, torch.device("cuda"), folds, rows, meta)
    print(json.dumps(summary["by_kind"], indent=2))


if __name__ == "__main__":
    main()
