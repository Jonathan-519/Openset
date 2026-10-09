#!/usr/bin/env python3
"""Replay the published H02 scores through the extracted calibration closure.

The large, immutable golden scores/predictions stay outside this repository.
Their exact Git blob and SHA256 identities are recorded in h02_expected.json.
This validates algorithms and frozen results; it does not run an image model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosieve import calibration as h02
from taxosieve import d05_calibration as d05


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def download_golden(directory, expected_path=None):
    """Fetch only pinned historical inputs; never replace conflicting files."""
    directory = Path(directory)
    expected = _json(expected_path or ROOT / "reproducibility/h02_expected.json")
    if (expected.get("repository") != "Jonathan-519/Openset"
            or expected.get("source_commit") != "abe3a982c4a6236cfbd57d7ca4169a6f5117950f"):
        raise ValueError("Golden download requires the reviewed repository and source commit")
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name = item["local_name"]
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Invalid golden filename")
        path = directory / name
        if path.is_symlink():
            raise ValueError("Golden inputs must be regular files")
        if path.exists():
            content = path.read_bytes()
        else:
            # Construct the immutable raw URL from the audited path instead of
            # following a branch or accepting a mutable download destination.
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid historical repository path")
            url = "https://raw.githubusercontent.com/{}/{}/{}".format(
                expected["repository"], expected["source_commit"], relative.as_posix())
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read(item["size_bytes"] + 1)
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if (len(content) != item["size_bytes"] or actual != item["git_blob_sha"]
                or hashlib.sha256(content).hexdigest() != item["sha256"]):
            raise ValueError("Golden input failed byte/SHA verification: " + name)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
        return name

    with ThreadPoolExecutor(max_workers=4) as pool:
        names = list(pool.map(fetch, expected["files"]))
    print("Verified {} immutable golden input files.".format(len(names)), flush=True)
    return names


def _equal(actual, expected, description):
    # Compact JSON also catches numeric serialization changes hidden by Python's
    # equality (for example, 0 versus 0.0). Ordering of mapping keys is irrelevant.
    if d05._hash(actual) != d05._hash(expected):
        raise AssertionError(description + " differs from the original frozen result")


def _by_split(records):
    groups = {}
    for record in records:
        groups.setdefault(record["split"], []).append(record)
    return groups


def _dev_groups(records):
    return [[r for r in records if r["status"] == status] for status in d05.base.STATUSES]


def _input_bytes(golden_dir, expected):
    receipts = []
    for entry in expected["files"]:
        path = golden_dir / entry["local_name"]
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if (blob != entry["git_blob_sha"] or sha256 != entry["sha256"]
                or len(data) != entry["size_bytes"]):
            raise AssertionError("Golden input changed: " + entry["local_name"])
        receipts.append(dict(file=entry["local_name"], sha256=sha256, git_blob_sha=blob,
                             size_bytes=len(data), source_url=entry["source_url"]))
    return receipts


def verify(golden_dir, *, expected_path=None, output=None):
    """Verify all frozen inputs, augmentation, routers, predictions and 11 folds."""
    started = time.monotonic()
    golden_dir = Path(golden_dir).resolve()
    expected = _json(expected_path or ROOT / "reproducibility" / "h02_expected.json")
    receipts = _input_bytes(golden_dir, expected)
    read = lambda name: _json(golden_dir / name)
    lines = lambda name: _jsonl(golden_dir / name)
    meta = expected["meta"]
    reference_router, d05_router, h02_router = (read(name) for name in
        ("c00_router.json", "d05_router.json", "h02_router.json"))
    for name, golden in (("reference_router.json", reference_router),
                         ("d05_router.json", d05_router), ("h02_router.json", h02_router)):
        _equal(_json(ROOT / "reproducibility" / name), golden, "Preserved " + name)
    d05.validate_router(d05_router, meta)
    h02.validate_router(h02_router, meta)
    c00_dev, d05_dev, dev, test = (lines(name) for name in
        ("c00_dev_scores.jsonl", "d05_dev_scores.jsonl", "h02_dev_scores.jsonl", "h02_test_scores.jsonl"))
    augmentation = {}
    for stage, raw, frozen in (("dev", d05_dev, dev), ("test", lines("d05_test_scores.jsonl"), test)):
        groups = _by_split(raw)
        source_hash = d05._hash(groups)
        augmented = h02.augment_scores(groups, meta)
        _equal(d05._hash(groups), source_hash, stage + " nonmutating augmentation")
        _equal(augmented, _by_split(frozen), stage + " complete H02 augmented score records")
        augmentation[stage] = dict(record_count=len(frozen), exact=True)
    reference_settings = read("c00_training_config.json")["calibration"]
    _equal(reference_settings, expected["reference_calibration_settings"], "C00 calibration settings")
    reference = dict(records=c00_dev, router=reference_router, calibration_settings=reference_settings)
    baseline = dict(records=d05_dev, router=d05_router)
    fitted_d05, _ = d05.fit_router(*_dev_groups(d05_dev), meta, d05_router["settings"], "global")
    _equal(fitted_d05, d05_router, "Complete D05 global router")
    fitted, fit_report = h02.fit_router(*_dev_groups(dev), meta, h02_router["settings"], "staged",
                                     reference_records=reference, d05_records=baseline)
    _equal(fitted, h02_router, "Complete H02 staged router")
    _equal(fit_report, read("h02_dev_report.json")["calibration_diagnostics"], "Complete DEV fit diagnostics")
    _equal(dict(root=fitted["root_threshold"], leaf=fitted["leaf_threshold"]),
           expected["thresholds"], "Both H02 thresholds")
    stages = {}
    for stage, scores in (("dev", dev), ("test", test)):
        predictions = h02.decode_records(scores, fitted, meta)
        # The historical artifact writer marks repeated image contents. The
        # decoder itself intentionally returns the undecorated original API.
        seen = set()
        for row in predictions:
            identity = d05.base._digest(row)
            row["evaluation_weight"] = int(identity not in seen)
            seen.add(identity)
        original_predictions = lines("h02_" + stage + "_predictions.jsonl")
        _equal(predictions, original_predictions, stage + " every complete prediction record")
        report = d05.base.evaluate_records(predictions, meta)
        summary = read("h02_" + stage + "_summary.json")
        _equal(report, {key: summary[key] for key in report}, stage + " evaluation including deduplication")
        _equal(report["counts"], expected[stage]["counts"], stage + " expected counts")
        stages[stage] = dict(record_count=len(predictions), unique_image_count=report["unique_image_count"],
                             duplicate_record_count=report["duplicate_record_count"],
                             complete_prediction_records_exact=True, counts=report["counts"],
                             metrics=report["metrics"], targets_passed=report["targets_passed"])
    print("Frozen routers, both thresholds and all DEV/TEST prediction records match; replaying conditional OOF.", flush=True)
    audit = h02.crossfit_audit(*_dev_groups(dev), meta, h02_router["settings"], "staged",
                              reference_records=reference, d05_records=baseline)
    original_audit = read("h02_crossfit_audit.json")
    _equal(audit, original_audit, "Complete conditional OOF report and all fold predictions")
    ids = [d05.base._digest(row) for row in audit["predictions"]]
    if len(ids) != len(set(ids)) or not audit["complete"]:
        raise AssertionError("OOF images are missing or repeated")
    for fold in audit["folds"]:
        fit, held = set(fold["fit_image_sha256"]), set(fold["held_image_sha256"])
        if fit & held or any(set(fold[key]) != fit for key in
            ("reference_fit_image_sha256", "d05_fit_image_sha256", "target_fit_image_sha256")):
            raise AssertionError("OOF fit/held identity isolation changed")
    report = dict(schema_version="h02_replay_verification_v1", passed=True,
        repository=expected["repository"], source_commit=expected["source_commit"],
        validation_scope=expected["validation_scope"], image_forward_validated=False, training_validated=False,
        elapsed_seconds=round(time.monotonic() - started, 3), verified_inputs=receipts,
        augmentation=augmentation, complete_d05_router_exact=True, complete_h02_router_exact=True,
        complete_dev_fit_diagnostics_exact=True, thresholds=expected["thresholds"], hashes=expected["hashes"],
        stages=stages, oof=dict(complete_report_exact=True, fold_count=len(audit["folds"]),
            evaluated_image_count=len(ids), every_image_held_once=True, fit_held_disjoint=True,
            all_three_routers_fit_only=True, counts=audit["counts"],
            d05_counts=audit["known_recovery"]["counts"]["d05"],
            historical_gates_passed=audit["passed"], historical_recovery_passed=audit["recovery_passed"]))
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", required=True, type=Path,
                        help="Directory containing the immutable golden files listed in h02_expected.json")
    parser.add_argument("--expected", type=Path, default=None)
    parser.add_argument("--download", action="store_true",
                        help="Fetch missing golden files from the pinned original commit and verify both hashes")
    parser.add_argument("--output", type=Path, default=None, help="Optional compact verification report")
    args = parser.parse_args()
    if args.download:
        download_golden(args.golden_dir, expected_path=args.expected)
    report = verify(args.golden_dir, expected_path=args.expected, output=args.output)
    print(json.dumps({key: report[key] for key in ("passed", "elapsed_seconds", "thresholds", "hashes", "stages", "oof")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
