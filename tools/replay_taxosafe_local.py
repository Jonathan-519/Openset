"""Audit local calibration from saved geometry scores without images/GPU.

This produces diagnostic reports, not deployable fit/checkpoint receipts.
TEST is optional and is opened only after the DEV selection is written/frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosafe_geometry import calibration as geometry
from taxosafe_geometry import local, protocol
from taxosafe_support import calibration as base


def _records(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay(source, reference_config, output, config=None, evaluate_test=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    reference_config = Path(reference_config).resolve()
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a new output directory outside the source geometry run")
    cfg = protocol.effective_config(config or ROOT /
        "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_local.yml")
    if cfg["calibration"].get("decoder") != local.DECODER:
        raise ValueError("Replay requires a local_guarded configuration")
    binding = protocol.read_json(source / "geometry/source_binding.json")
    if protocol.file_hash(reference_config) != binding["receipt_sha256"]["training/config.json"]:
        raise ValueError("Reference config differs from the geometry source binding")
    source_cfg = protocol.read_json(reference_config)
    old_router = protocol.read_json(source / "calibration/router.json")
    baseline, meta = old_router["baseline_router"], old_router["meta"]
    if old_router.get("baseline_router_sha256") != geometry._hash(baseline):
        raise ValueError("Geometry baseline binding mismatch")
    report = protocol.read_json(source / "geometry/fit_report.json")
    if (report.get("fit_splits") != ["train"] or report.get("test_used_for_fitting") is not False or
            report.get("unknown_images_used_for_fitting") is not False):
        raise ValueError("Expected TRAIN-only geometry statistics")
    rows = _records(source / "calibration/development_scores.jsonl")
    groups = [[r for r in rows if r["status"] == s] for s in ("known", "intra", "extra")]
    if sum(map(len, groups)) != len(rows):
        raise ValueError("Unexpected DEV status")
    options = dict(cfg["calibration"], baseline_calibration=source_cfg["calibration"])
    router = local.calibrate(*groups, baseline, meta, options)
    output.mkdir(parents=True)
    protocol.write_json(output / "router.json", router)
    code = sorted((ROOT / "taxosafe_geometry").glob("*.py")) + [Path(__file__)]
    freeze = dict(schema="local_saved_scores_review_v1", diagnostic_only=True,
                  test_used_for_fitting=False, test_opened=False,
                  config=cfg, code_sha256={str(p.relative_to(ROOT)): protocol.file_hash(p) for p in code},
                  source_sha256={str(p.relative_to(source)): protocol.file_hash(p) for p in (
                      source / "geometry/source_binding.json", source / "geometry/fit_report.json",
                      source / "calibration/router.json", source / "calibration/development_scores.jsonl")},
                  router_sha256=protocol.file_hash(output / "router.json"))
    # Commit the fitted operating point to disk before any TEST file is read.
    protocol.write_json(output / "selection_frozen.json", freeze)
    summary = dict(development_baseline=router["baseline_validation_report"],
                   development_selected=router["validation_report"],
                   enabled_actions=router["enabled_actions"], rejected_actions=router["rejected_actions"],
                   source_safeguard=router["source_loo"]["safeguard"],
                   test_used_for_fitting=False, diagnostic_only=True)
    if evaluate_test:
        tests = _records(source / "test/predictions.jsonl")
        if any(r.get("split") != "test_" + r["status"] for r in tests):
            raise ValueError("Expected only TEST records for held-out replay")
        if set(router["fit_image_sha256"]) & {base._digest(r) for r in tests}:
            raise ValueError("DEV/TEST image overlap")
        before, after = base.apply_router(tests, baseline, meta), local.apply_router(tests, router, meta)
        archived = {base._digest(r): r for r in _records(source / "test/baseline_predictions.jsonl")}
        for r in before:
            if base._digest(r) not in archived or any(r[k] != archived[base._digest(r)][k] for k in (
                    "prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf")):
                raise ValueError("Archived reference TEST decisions do not reproduce")
        summary["test_baseline"] = base.evaluate_records(before, meta)
        summary["test_selected"] = base.evaluate_records(after, meta)
        summary["test_component_scores"] = local.score_diagnostics(tests)
        summary["test_scores_sha256"] = protocol.file_hash(source / "test/predictions.jsonl")
        unique_before, unique_after = base.unique_records(before), base.unique_records(after)
        summary["test_known_protection"] = {
            "lost_baseline_correct": sum(b["status"] == "known" and b["prediction_type"] == "known" and
                b["leaf"] == b["true_leaf"] and b["parent"] == b["true_parent"] and a["prediction_type"] != "known"
                for b, a in zip(unique_before, unique_after)),
            "scope": "observed replay only; DEV protection does not guarantee TEST preservation"}
        protocol.write_records(output / "test_predictions.jsonl", after)
        if protocol.file_hash(output / "router.json") != freeze["router_sha256"]:
            raise ValueError("TEST must not change the frozen router")
    protocol.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run-dir", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    summary = replay(args.geometry_run_dir, args.reference_config, args.output_dir, args.config, args.evaluate_test)
    print(json.dumps({k: v.get("metrics") for k, v in summary.items() if isinstance(v, dict) and "metrics" in v}, indent=2))
    print("Saved score audit: " + str(args.output_dir))


if __name__ == "__main__":
    main()
