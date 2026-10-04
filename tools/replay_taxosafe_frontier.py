"""DEV-only exploratory replay from ParentRisk text diagnostics.

This is not a fit/calibration completion receipt or a model backup. It reads
only the named TRAIN/DEV metadata and DEV score files, never TEST. Formal image
execution uses ``python -m taxosafe_frontier`` in a fresh run directory.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosafe_frontier import calibration, protocol
from taxosafe_parentrisk import protocol as old_protocol
from taxosafe_support.calibration import _digest


def _read(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Expected ordinary local diagnostic file: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(directory, receipt, key, filename):
    item = receipt.get(key)
    path = Path(directory) / filename
    if (not isinstance(item, dict) or item.get("path") != filename
            or path.is_symlink() or not path.is_file()
            or protocol.file_hash(path) != item.get("sha256")):
        raise ValueError("Review text artifact hash/path mismatch: " + filename)
    return path


def replay(source, output):
    source, output = Path(source).absolute(), Path(output).absolute()
    if any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("Source review must not traverse a symlink")
    source = source.resolve()
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a new output directory separate from the source review")
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Replay output must not traverse a symlink")
    fitted = _read(source / "parentrisk/completed.json")
    receipt = _read(source / "calibration/completed.json")
    if (fitted.get("schema_version") != old_protocol.SCHEMA_VERSION
            or receipt.get("schema_version") != old_protocol.SCHEMA_VERSION
            or receipt.get("fit_splits") != ["val_known", "val_intra", "val_extra"]
            or receipt.get("test_used_for_fitting") is not False
            or receipt.get("signature") != fitted.get("signature")
            or receipt.get("fit_receipt_sha256") != protocol.file_hash(source / "parentrisk/completed.json")
            or receipt.get("model_sha256") != fitted.get("model", {}).get("sha256")
            or receipt.get("source_binding") != fitted.get("source_binding")):
        raise ValueError("Expected a consistent original ParentRisk TRAIN/DEV text receipt chain")
    cal = source / "calibration"
    old_router_path = _artifact(cal, receipt, "router", "router.json")
    scores_path = _artifact(cal, receipt, "scores", "development_scores.jsonl")
    old_router = _read(old_router_path)
    if old_router.get("signature") != receipt["signature"]:
        raise ValueError("Original router signature differs from the DEV receipt")
    records = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    settings = old_router["settings"]
    options = {key: settings[key] for key in ("outer_folds", "inner_folds", "min_rule_sources", "seed", "baseline_calibration")}
    groups = [[row for row in records if row.get("status") == status] for status in ("known", "intra", "extra")]
    if sum(map(len, groups)) != len(records):
        raise ValueError("Unexpected DEV status")
    code_signature = protocol.signature({"replay_options": options}, receipt["source_binding"])
    tool_hash = protocol.file_hash(Path(__file__))
    result = calibration.calibrate(*groups, old_router["baseline_router"], old_router["meta"], options)
    protocol.require_signature(code_signature, protocol.signature({"replay_options": options}, receipt["source_binding"]))
    if protocol.file_hash(Path(__file__)) != tool_hash:
        raise ValueError("Replay tool changed during execution")
    provenance = {
        "kind": "exploratory_dev_text_replay_not_production_receipt",
        "original_method": old_protocol.SCHEMA_VERSION,
        "new_method": protocol.SCHEMA_VERSION,
        "source_development_scores_sha256": protocol.file_hash(scores_path),
        "source_calibration_router_sha256": protocol.file_hash(old_router_path),
        "source_fit_receipt_sha256": protocol.file_hash(source / "parentrisk/completed.json"),
        "source_signature": receipt["signature"],
        "new_code_sha256": code_signature["code"],
        "replay_tool_sha256": tool_hash,
        "dev_unique_images": len({_digest(row) for row in records}),
        "test_files_opened": False,
        "image_pipeline_executed": False,
        "checkpoint_and_cache_binary_verified": False,
        "validation_scope": "exploratory_postprocessor_on_previously_used_development",
        "development_reused_for_method_design": True,
        "independent_model_level_validation": False,
        "confirmatory_validation": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    protocol.write_json(output / "replay_provenance.json", provenance)
    protocol.write_json(output / "router.json", result)
    summary = {
        "provenance": provenance,
        "selection_status": result["selection_status"],
        "baseline_fallback": result["baseline_fallback"],
        "research_targets_passed": result["targets_passed"],
        "reject_rules": result["reject_rules"],
        "development_baseline": result["baseline_validation_report"],
        "development_selected": result["validation_report"],
        "full_fit_audit": result["full_fit_audit"],
        "inner_selection": result["inner_selection"],
        "outer_status": result["outer_audit"]["status"],
        "outer_report": result["outer_audit"]["report"],
        "outer_output_used_for_selection": result["outer_audit"]["output_used_for_selection"],
    }
    protocol.write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "selection_status": summary["selection_status"],
                      "baseline_fallback": summary["baseline_fallback"],
                      "research_targets_passed": summary["research_targets_passed"],
                      "production_receipt": False}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(p.is_symlink() for p in (args.review_run_dir.absolute(), *args.review_run_dir.absolute().parents)):
        raise ValueError("Source review must not traverse a symlink")
    replay(args.review_run_dir, args.output)


if __name__ == "__main__":
    main()
