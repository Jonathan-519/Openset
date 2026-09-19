"""Upgrade an existing TaxoLocal-v2 router to v2.1 without retraining."""

import argparse
import hashlib
import json
import os

import yaml

from metrics_open import evaluate_open_set
from taxolocal_v21_router import apply_router, upgrade_router


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=yaml.SafeLoader)
    if not isinstance(value, dict):
        raise ValueError("The YAML root must be a mapping")
    return value, path


def resolve_run_dir(cfg, trial, project_root, explicit_run_dir=None):
    if explicit_run_dir:
        return os.path.abspath(explicit_run_dir)
    return os.path.join(
        project_root, "runs", cfg["data"]["name"], cfg["model"]["arch"],
        cfg["exp"], "trial_{}".format(trial),
    )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path, records, drop_vector_fields=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            output = dict(record)
            if drop_vector_fields:
                output.pop("parent_cosine", None)
                output.pop("leaf_cosine", None)
                output.pop("image_feature", None)
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError("Development predictions are empty: {}".format(path))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--base-router", default=None)
    parser.add_argument("--development-predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_yaml(args.config)
    os.chdir(PROJECT_ROOT)
    run_dir = resolve_run_dir(cfg, args.trial, PROJECT_ROOT, args.run_dir)
    base_router_path = os.path.abspath(
        args.base_router
        or os.path.join(run_dir, "router_v2", "router.json")
    )
    development_path = os.path.abspath(
        args.development_predictions
        or os.path.join(
            run_dir, "router_v2", "development_predictions.jsonl"
        )
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(run_dir, "router_v21")
    )
    router_path = os.path.join(output_dir, "router.json")
    metrics_path = os.path.join(output_dir, "development_metrics.json")
    predictions_path = os.path.join(
        output_dir, "development_predictions.jsonl"
    )
    if not args.overwrite and any(os.path.exists(path) for path in (
        router_path, metrics_path, predictions_path
    )):
        raise FileExistsError("TaxoLocal-v2.1 router outputs already exist")
    with open(base_router_path, "r", encoding="utf-8") as stream:
        base_router = json.load(stream)
    metadata = base_router.get("metadata", {})
    if metadata.get("test_data_loaded") is not False:
        raise RuntimeError("The source router does not prove test isolation")
    if any("test" in split for split in metadata.get(
        "development_splits", []
    )):
        raise RuntimeError("The source router used a test calibration split")

    development_records = read_jsonl(development_path)
    router = upgrade_router(
        base_router=base_router,
        development_records=development_records,
        settings=cfg.get("router_v21", {}),
    )
    router["metadata"] = dict(metadata)
    router["metadata"].update({
        "source_router": base_router_path,
        "source_router_sha256": sha256_file(base_router_path),
        "source_development_predictions": development_path,
        "source_development_predictions_sha256": sha256_file(
            development_path
        ),
        "test_data_loaded": False,
        "checkpoint_updated": False,
        "training_repeated": False,
    })
    predictions = apply_router(
        development_records,
        router,
        router["parent_names"],
        router["leaf_names"],
        scores_already_added=True,
    )
    metrics = evaluate_open_set(
        [row for row in predictions if row["status"] == "known"],
        [row for row in predictions if row["status"] == "intra"],
        [row for row in predictions if row["status"] == "extra"],
    )
    metrics["metadata"] = {
        "method": router["method"],
        "source_router": base_router_path,
        "test_data_loaded": False,
        "checkpoint_updated": False,
    }
    write_json(router_path, router)
    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, predictions, drop_vector_fields=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
