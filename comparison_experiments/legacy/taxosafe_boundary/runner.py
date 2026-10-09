"""Isolated boundary experiments, with DEV frozen before original TEST features.

Failed research gates never block TEST. A technical stage failure is recorded
without invented metrics; only its dependencies are blocked. Changed source,
code, configuration or already verified artifacts invalidate the suite.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from taxosafe_routealign.runner import (
    _destinations, _device_check, _regular, _manifest, _stamp,
)
from . import protocol
from .importer import inspect_d05 as _source

STAGE_MARKER = "stage_binding.json"
ARM_STAGES = ("training", "calibration", "test")
CACHE_STAGES = ("train", "development", "test")
WORKER_STAGES = tuple("cache_" + name for name in CACHE_STAGES) + ARM_STAGES


def _directory(suite, arm_id, stage):
    return Path(suite) / ("cache" if arm_id is None else "arms") / (stage[6:] if arm_id is None else arm_id) / ("" if arm_id is None else stage)


def _dependency(arm):
    # Reused weights need their source training, independently of its calibration.
    return arm.get("weight_source")


def _initialize(suite, cfg, source, device):
    suite.mkdir(parents=True, exist_ok=False)
    protocol.write_json(suite / "config.json", cfg)
    protocol.write_json(suite / "source_binding.json", source["binding"])
    value = {"schema_version": protocol.SCHEMA_VERSION, "signature": protocol.signature(cfg, source["binding"]),
             "config_sha256": protocol.object_hash(cfg), "source_binding": source["binding"],
             "arm_ids": [arm["id"] for arm in cfg["arms"]], "device": device,
             "runtime": dict(_device_check(device), python=sys.version.split()[0]),
             "created_at_utc": _stamp(), "evaluation_order": "train_cache_then_dev_cache_then_all_dev_then_freeze_then_test_cache_then_all_test",
             "test_on_failed_research_gate": True, "test_used_for_selection": False}
    protocol.write_json(suite / "snapshot.json", value)
    return value


def _verify_snapshot(suite, cfg, source):
    suite = Path(suite)
    snapshot = protocol.read_json(_regular(suite / "snapshot.json"))
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("signature") != protocol.signature(cfg, source["binding"])
            or snapshot.get("config_sha256") != protocol.object_hash(cfg)
            or snapshot.get("source_binding") != source["binding"]
            or snapshot.get("arm_ids") != [arm["id"] for arm in cfg["arms"]]
            or protocol.read_json(_regular(suite / "config.json")) != cfg
            or protocol.read_json(_regular(suite / "source_binding.json")) != source["binding"]):
        raise ValueError("Suite source/configuration/code snapshot changed; use a fresh suite")
    return snapshot


def _receipt(directory, snapshot, arm_id, stage):
    receipt = protocol.read_json(_regular(directory / "completed.json"))
    expected_stage = stage[6:] if arm_id is None else stage
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != protocol.SCHEMA_VERSION or receipt.get("stage") != expected_stage
            or receipt.get("signature") != snapshot["signature"]
            or receipt.get("source_binding") != snapshot["source_binding"]
            or (arm_id is not None and receipt.get("arm_id") != arm_id)):
        raise ValueError("Stage receipt schema/source/signature identity mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Stage receipt must declare its actual artifacts")
    for item in artifacts.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Invalid stage artifact descriptor")
        name = item["path"]
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Stage artifacts must be local regular filenames")
        if protocol.file_hash(_regular(directory / name)) != item["sha256"]:
            raise ValueError("Stage artifact hash mismatch: " + name)
    return receipt


def _complete_stage(suite, arm_id, stage, snapshot, started_at_utc=None, elapsed_seconds=None):
    directory = _directory(suite, arm_id, stage)
    marker = directory / STAGE_MARKER
    if marker.exists():
        raise ValueError("Cannot overwrite a verified stage binding")
    _receipt(directory, snapshot, arm_id, stage)
    protocol.write_json(marker, {"schema_version": protocol.SCHEMA_VERSION, "arm_id": arm_id,
        "stage": stage, "suite_signature": snapshot["signature"],
        "snapshot_sha256": protocol.file_hash(Path(suite) / "snapshot.json"), "files": _manifest(directory),
        "started_at_utc": started_at_utc, "completed_at_utc": _stamp(), "elapsed_seconds": elapsed_seconds})


def _verify_stage(suite, arm_id, stage, snapshot):
    directory = _directory(suite, arm_id, stage)
    marker = protocol.read_json(_regular(directory / STAGE_MARKER))
    if (not isinstance(marker, dict) or marker.get("schema_version") != protocol.SCHEMA_VERSION or marker.get("arm_id") != arm_id
            or marker.get("stage") != stage or marker.get("suite_signature") != snapshot["signature"]
            or marker.get("snapshot_sha256") != protocol.file_hash(Path(suite) / "snapshot.json")
            or marker.get("files") != _manifest(directory)):
        raise ValueError("Completed stage artifacts changed: " + str(directory))
    for key in ("started_at_utc", "completed_at_utc"):
        value = marker.get(key)
        if value is None and key == "started_at_utc":
            continue
        if not isinstance(value, str) or datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Stage timestamp must include its timezone: " + key)
    elapsed = marker.get("elapsed_seconds")
    if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                                or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Stage elapsed seconds must be a finite nonnegative value")
    return _receipt(directory, snapshot, arm_id, stage)


def _failure(suite, arm_id, stage, message, stderr=None):
    directory = Path(suite) / "arms" / arm_id if arm_id is not None else _directory(suite, None, stage)
    directory.mkdir(parents=True, exist_ok=True)
    value = {"arm_id": arm_id, "stage": stage, "error": str(message), "technical_failure": True,
             "research_gate_failure": False, "test_blocked": stage not in ("test", "cache_test"),
             "created_at_utc": _stamp(), "stderr_log": None if stderr is None else str(Path(stderr).relative_to(suite))}
    protocol.write_json(directory / "failure.json", value)
    return value


def _resume_audit(suite, cfg, source):
    snapshot = _verify_snapshot(suite, cfg, source)
    for failure in Path(suite).glob("cache/*/failure.json"):
        raise ValueError("Cannot resume a technically failed cache; preserve it and use a fresh suite: " + str(failure))
    for failure in Path(suite).glob("arms/*/failure.json"):
        raise ValueError("Cannot resume a technically failed arm; preserve it and use a fresh suite: " + str(failure))
    for stage in CACHE_STAGES:
        if _directory(suite, None, "cache_" + stage).exists():
            _verify_stage(suite, None, "cache_" + stage, snapshot)
    for arm in cfg["arms"]:
        for stage in ARM_STAGES:
            if not _directory(suite, arm["id"], stage).exists():
                continue
            _verify_stage(suite, arm["id"], stage, snapshot)
            for cache in ("train", "development") + (("test",) if stage == "test" else ()):
                _verify_stage(suite, None, "cache_" + cache, snapshot)
            if stage != "training":
                _verify_stage(suite, arm["id"], "training", snapshot)
            if stage == "test":
                _verify_stage(suite, arm["id"], "calibration", snapshot)
            dependency = _dependency(arm)
            if dependency:
                _verify_stage(suite, dependency, "training", snapshot)
    if (Path(suite) / "cache/test").exists() or any(Path(suite).glob("arms/*/test")):
        if not (Path(suite) / "dev_selection.json").is_file():
            raise ValueError("TEST cache/stages require an already frozen DEV selection")
    return snapshot


def preflight(cfg, discovery_directory, suite_directory, device="cpu", resume=False):
    """Validate parent bytes and cached TRAIN/DEV metadata without loading tensors."""
    protocol.validate_config(cfg)
    if device != "cpu":
        raise ValueError("Boundary uses existing CPU feature caches; device must be cpu")
    discovery, suite = _destinations(discovery_directory, suite_directory)
    source, runtime = _source(discovery), _device_check(device)
    _destinations(source["reference"]["directory"], suite)
    if resume:
        if _resume_audit(suite, cfg, source)["device"] != device:
            raise ValueError("Device differs from the immutable suite snapshot")
    elif suite.exists():
        raise ValueError("Use a fresh absent suite directory; preserve existing runs")
    audits = {key: value for receipt in source["caches"].values() for key, value in receipt["audit"].items()}
    return {"schema_version": protocol.SCHEMA_VERSION, "source_binding": source["binding"],
        "signature": protocol.signature(cfg, source["binding"]), "runtime": runtime,
        "locked_split_counts": {key: value["unique_image_count"] for key, value in audits.items()},
        "checkpoint_tensors_loaded": False, "model_forward_performed": False,
        "test_predictions_read": False, "test_cache_opened": False, "destination_created": False}


def worker(suite_directory, arm_id, stage, device):
    from . import backend, reporting
    started_at_utc, start = _stamp(), time.monotonic()
    suite = Path(suite_directory).resolve()
    cfg = protocol.read_json(_regular(suite / "config.json"))
    recorded = protocol.read_json(_regular(suite / "source_binding.json"))
    snapshot = _verify_snapshot(suite, cfg, _source(recorded["directory"]))
    if snapshot["device"] != device:
        raise ValueError("Device differs from the immutable suite snapshot")
    _device_check(device)
    if stage not in WORKER_STAGES or ((arm_id is None) != stage.startswith("cache_")):
        raise ValueError("Unknown cache/arm worker stage")
    arm = next((a for a in cfg["arms"] if a["id"] == arm_id), None)
    if arm_id is not None and arm is None:
        raise ValueError("Unknown boundary arm")
    if _directory(suite, arm_id, stage).exists():
        raise ValueError("Stage already exists; resume only verified complete stages")
    if stage in ("cache_test", "test"):
        _regular(suite / "dev_selection.json")
        reporting.freeze_dev_selection(suite)
    if stage.startswith("cache_"):
        backend.prepare_cache(suite, stage[6:], device)
    else:
        for cache in ("train", "development") + (("test",) if stage == "test" else ()):
            _verify_stage(suite, None, "cache_" + cache, snapshot)
        dependency = _dependency(arm)
        if dependency:
            _verify_stage(suite, dependency, "training", snapshot)
        if stage != "training":
            _verify_stage(suite, arm_id, "training", snapshot)
        if stage == "test":
            _verify_stage(suite, arm_id, "calibration", snapshot)
        {"training": backend.fit_arm, "calibration": backend.calibrate_arm, "test": backend.test_arm}[stage](suite, arm_id, device)
    _verify_snapshot(suite, cfg, _source(recorded["directory"]))
    _complete_stage(suite, arm_id, stage, snapshot, started_at_utc, time.monotonic() - start)
    if stage in ("calibration", "test"):
        reporting._stage(suite, arm_id, stage, snapshot)


def _launch_stage(suite, arm_id, stage, device):
    logs = Path(suite) / "logs" / ("cache" if arm_id is None else arm_id)
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
    command = [sys.executable, "-u", "-m", "taxosafe_boundary", "--worker-stage", stage,
               "--suite-dir", str(suite), "--device", device]
    if arm_id is not None:
        command.extend(("--arm", arm_id))
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=str(protocol.PROJECT_ROOT), stdout=out, stderr=err, check=False)
    return result.returncode, stdout, stderr


def _execute(suite, cfg, source_directory, arm_id, stage, device, resume=False):
    snapshot = _verify_snapshot(suite, cfg, _source(source_directory))
    if _directory(suite, arm_id, stage).exists():
        if not resume:
            raise ValueError("Stage unexpectedly exists; do not overwrite artifacts")
        _verify_stage(suite, arm_id, stage, snapshot)
        return True
    label = "cache" if arm_id is None else arm_id
    print("[{}] {} {} started".format(_stamp(), label, stage), flush=True)
    try:
        returncode, stdout, stderr = _launch_stage(suite, arm_id, stage, device)
    except OSError as error:
        _verify_snapshot(suite, cfg, _source(source_directory))
        _failure(suite, arm_id, stage, "Cannot launch isolated stage: " + str(error))
        return False
    _verify_snapshot(suite, cfg, _source(source_directory))
    if returncode:
        _failure(suite, arm_id, stage, "stage exit {}: {}".format(returncode,
            stderr.read_text(encoding="utf-8", errors="replace")[-6000:]), stderr)
        return False
    try:
        receipt = _verify_stage(suite, arm_id, stage, snapshot)
        if stage in ("calibration", "test"):
            from . import reporting
            reporting._stage(Path(suite), arm_id, stage, snapshot)
    except (ValueError, OSError, KeyError, TypeError) as error:
        _failure(suite, arm_id, stage, "Stage did not produce a verified completion: " + str(error), stderr)
        return False
    print("[{}] {} {} completed; targets_passed={}".format(_stamp(), label, stage, receipt.get("targets_passed")), flush=True)
    return True


def execute_suite(cfg, discovery_directory, suite_directory, device="cpu", resume=False, run_preflight=True):
    from . import reporting
    reference, suite = _destinations(discovery_directory, suite_directory)
    protocol.validate_config(cfg)
    if device != "cpu":
        raise ValueError("Boundary uses cached features on CPU")
    if run_preflight:
        preflight(cfg, reference, suite, device, resume)
    source = _source(reference)
    _destinations(source["reference"]["directory"], suite)
    snapshot = _resume_audit(suite, cfg, source) if resume else _initialize(suite, cfg, source, device)
    if snapshot["device"] != device:
        raise ValueError("Device differs from the immutable suite snapshot")
    calibrated, tested = [], []
    with protocol.run_lock(suite):
        caches = {stage: _execute(suite, cfg, reference, None, "cache_" + stage, device, resume)
                  for stage in ("train", "development")}
        for arm in cfg["arms"]:
            arm_id, dependency = arm["id"], _dependency(arm)
            if not all(caches.values()):
                _failure(suite, arm_id, "dependency", "Required cache failed: " + ", ".join(k for k, ok in caches.items() if not ok))
                continue
            if dependency and not (_directory(suite, dependency, "training") / STAGE_MARKER).is_file():
                _failure(suite, arm_id, "dependency", "Required training is unavailable: " + dependency)
                continue
            if not _execute(suite, cfg, reference, arm_id, "training", device, resume):
                continue
            if _execute(suite, cfg, reference, arm_id, "calibration", device, resume):
                calibrated.append(arm_id)
        reporting.freeze_dev_selection(suite)
        frozen_hash = protocol.file_hash(suite / "dev_selection.json")
        if calibrated:
            test_cache = _execute(suite, cfg, reference, None, "cache_test", device, resume)
            if protocol.file_hash(suite / "dev_selection.json") != frozen_hash:
                raise ValueError("TEST cache changed the frozen DEV selection")
            for arm_id in calibrated:
                if not test_cache:
                    _failure(suite, arm_id, "test", "Required TEST cache failed; no test metrics were produced")
                elif _execute(suite, cfg, reference, arm_id, "test", device, resume):
                    tested.append(arm_id)
                if protocol.file_hash(suite / "dev_selection.json") != frozen_hash:
                    raise ValueError("TEST changed the frozen DEV selection")
        _verify_snapshot(suite, cfg, _source(reference))
        summary = reporting.summarize_suite(suite, phase="complete")
        protocol.write_json(suite / "suite_completed.json", {"schema_version": protocol.SCHEMA_VERSION,
            "signature": snapshot["signature"], "workflow_completed": True, "completed_at_utc": _stamp(),
            "dev_selection_sha256": frozen_hash, "test_used_for_selection": False,
            "all_valid_calibrations_test_attempted_regardless_of_gate": True,
            "completed_test_arms": tested, "completed_calibration_arms": calibrated,
            "technical_failure_arms": [a["id"] for a in cfg["arms"] if (suite / "arms" / a["id"] / "failure.json").exists()]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--discovery-run-dir", "--source-suite-dir", dest="discovery_run_dir",
                        default="runs/taxosafe_new/discovery/trial_1_20261005_175231")
    parser.add_argument("--run-dir", "--suite-dir", dest="suite_dir", type=Path,
                        help="Fresh output directory; defaults to boundary/trial_1_<timestamp>")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker-stage", choices=WORKER_STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_stage:
        if args.suite_dir is None or args.preflight or args.resume or (args.worker_stage.startswith("cache_") == bool(args.arm)):
            parser.error("Cache workers have no arm; arm workers require --arm; neither can resume/preflight")
        worker(args.suite_dir, args.arm, args.worker_stage, args.device)
        return
    if not args.discovery_run_dir:
        parser.error("--discovery-run-dir is required; choose the intact D05 discovery suite explicitly")
    if args.suite_dir is None:
        if args.resume:
            parser.error("--resume requires the explicit existing --run-dir")
        args.suite_dir = Path("runs/taxosafe_new/boundary") / ("trial_1_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    config_path = Path(args.config) if Path(args.config).is_file() else protocol.resolve(args.config)
    cfg = protocol.effective_config(config_path)
    source, suite = protocol.resolve(args.discovery_run_dir), protocol.resolve(str(args.suite_dir))
    if args.preflight:
        print(json.dumps(preflight(cfg, source, suite, args.device, args.resume), indent=2))
        return
    try:
        execute_suite(cfg, source, suite, args.device, args.resume)
    finally:
        if (suite / "snapshot.json").is_file():
            from tools.pack_taxosafe_boundary_review import pack
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            pack(suite, suite.with_name(suite.name + "_review_" + stamp + ".tar.gz"))


if __name__ == "__main__":
    main()
