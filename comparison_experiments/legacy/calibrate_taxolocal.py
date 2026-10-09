"""Fit the TaxoLocal router using development data only."""

import argparse
import json
import os

import torch

from metrics_open import evaluate_open_set
from taxolocal_router import apply_router, calibrate_router
from taxosafe_eval_utils import (
    collect_score_records,
    load_model_and_data,
    load_yaml,
    resolve_run_dir,
    sha256_file,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVELOPMENT_SPLITS = ("val_known", "dev_unknown", "val_extra")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_yaml(args.config)
    os.chdir(PROJECT_ROOT)
    run_dir = resolve_run_dir(cfg, args.trial, PROJECT_ROOT, args.run_dir)
    checkpoint = os.path.abspath(
        args.checkpoint or os.path.join(run_dir, "ckpt", "best.pth")
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(run_dir, "router")
    )
    router_path = os.path.join(output_dir, "router.json")
    metrics_path = os.path.join(output_dir, "development_metrics.json")
    predictions_path = os.path.join(output_dir, "development_predictions.jsonl")
    if not args.overwrite and any(
        os.path.exists(path)
        for path in (router_path, metrics_path, predictions_path)
    ):
        raise FileExistsError("Development router outputs already exist")
    if not torch.cuda.is_available():
        raise SystemExit("TaxoLocal ViT-B/16 calibration requires CUDA")
    device = torch.device("cuda")
    model, loaders, meta = load_model_and_data(
        cfg, DEVELOPMENT_SPLITS, checkpoint, device
    )
    unknown_template = cfg.get("open_treecut", {}).get(
        "unknown_template", "novel member of {}"
    )
    known = collect_score_records(
        model, loaders["val_known"], "known", meta, device,
        unknown_template=unknown_template,
    )
    novel = collect_score_records(
        model, loaders["dev_unknown"], "intra", meta, device,
        unknown_template=unknown_template,
    )
    extra = collect_score_records(
        model, loaders["val_extra"], "extra", meta, device,
        unknown_template=unknown_template,
    )
    router = calibrate_router(
        known, novel, extra, meta["parent_names"], cfg.get("router", {})
    )
    router["metadata"] = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "development_splits": list(DEVELOPMENT_SPLITS),
        "test_data_loaded": False,
    }
    predictions = apply_router(
        known + novel + extra,
        router,
        meta["parent_names"],
        meta["leaf_names"],
    )
    metrics = evaluate_open_set(
        [row for row in predictions if row["status"] == "known"],
        [row for row in predictions if row["status"] == "intra"],
        [row for row in predictions if row["status"] == "extra"],
    )
    write_json(router_path, router)
    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, predictions, drop_vector_fields=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
