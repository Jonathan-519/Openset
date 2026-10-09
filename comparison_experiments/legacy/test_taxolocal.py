"""Run the locked TaxoLocal test exactly once after development is frozen."""

import argparse
import json
import os

import torch

from metrics_open import evaluate_open_set
from taxolocal_router import apply_router
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
LOCKED_SPLITS = ("test_known", "test_unknown", "test_extra")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--router", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg, _ = load_yaml(args.config)
    os.chdir(PROJECT_ROOT)
    run_dir = resolve_run_dir(cfg, args.trial, PROJECT_ROOT, args.run_dir)
    checkpoint = os.path.abspath(
        args.checkpoint or os.path.join(run_dir, "ckpt", "best.pth")
    )
    router_path = os.path.abspath(
        args.router or os.path.join(run_dir, "router", "router.json")
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(run_dir, "locked_test")
    )
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    if os.path.exists(metrics_path) or os.path.exists(predictions_path):
        raise FileExistsError(
            "Locked test outputs already exist; refusing to overwrite"
        )
    with open(router_path, "r", encoding="utf-8") as stream:
        router = json.load(stream)
    checkpoint_hash = sha256_file(checkpoint)
    if router.get("metadata", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Router was fitted with a different checkpoint")
    if any(
        "test" in split
        for split in router.get("metadata", {}).get("development_splits", [])
    ):
        raise RuntimeError("Router metadata contains a test calibration split")
    if not torch.cuda.is_available():
        raise SystemExit("TaxoLocal ViT-B/16 testing requires CUDA")
    device = torch.device("cuda")
    model, loaders, meta = load_model_and_data(
        cfg, LOCKED_SPLITS, checkpoint, device
    )
    unknown_template = cfg.get("open_treecut", {}).get(
        "unknown_template", "novel member of {}"
    )
    known = collect_score_records(
        model, loaders["test_known"], "known", meta, device,
        unknown_template=unknown_template,
    )
    novel = collect_score_records(
        model, loaders["test_unknown"], "intra", meta, device,
        unknown_template=unknown_template,
    )
    extra = collect_score_records(
        model, loaders["test_extra"], "extra", meta, device,
        unknown_template=unknown_template,
    )
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
    metrics["metadata"] = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "router": router_path,
        "test_splits": list(LOCKED_SPLITS),
        "locked_test_overwrite_allowed": False,
    }
    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, predictions, drop_vector_fields=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
