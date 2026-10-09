"""Evaluate a frozen TaxoLocal-v2.1 router on the existing comparison split."""

import argparse
import json
import os

import torch

from metrics_open import evaluate_open_set
from taxolocal_v21_router import apply_router
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
TEST_SPLITS = ("test_known", "test_unknown", "test_extra")


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
        args.router or os.path.join(run_dir, "router_v21", "router.json")
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(run_dir, "test_v21_development")
    )
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    if os.path.exists(metrics_path) or os.path.exists(predictions_path):
        raise FileExistsError("TaxoLocal-v2.1 test outputs already exist")
    with open(router_path, "r", encoding="utf-8") as stream:
        router = json.load(stream)
    checkpoint_hash = sha256_file(checkpoint)
    metadata = router.get("metadata", {})
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Router was fitted with a different checkpoint")
    if metadata.get("test_data_loaded") is not False:
        raise RuntimeError("Router metadata does not prove test isolation")
    if any("test" in split for split in metadata.get(
        "development_splits", []
    )):
        raise RuntimeError("Router metadata contains a test calibration split")
    if not torch.cuda.is_available():
        raise SystemExit("TaxoLocal-v2.1 testing requires CUDA")

    device = torch.device("cuda")
    model, loaders, meta = load_model_and_data(
        cfg, TEST_SPLITS, checkpoint, device
    )
    unknown_template = cfg.get("open_treecut", {}).get(
        "unknown_template", "novel member of {}"
    )
    common = {
        "model": model,
        "hier_meta": meta,
        "device": device,
        "include_image_features": True,
        "unknown_template": unknown_template,
    }
    known = collect_score_records(
        data_loader=loaders["test_known"], status="known", **common
    )
    novel = collect_score_records(
        data_loader=loaders["test_unknown"], status="intra", **common
    )
    extra = collect_score_records(
        data_loader=loaders["test_extra"], status="extra", **common
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
        "test_splits": list(TEST_SPLITS),
        "protocol": "existing_test_previously_inspected_development_comparison",
        "new_blind_test_claim_allowed": False,
        "training_repeated": False,
    }
    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, predictions, drop_vector_fields=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
