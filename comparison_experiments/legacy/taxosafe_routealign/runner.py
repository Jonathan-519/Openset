"""Run every predefined arm; freeze DEV selection before evaluating every TEST.

Each stage runs in a fresh Python process to release accelerator allocations.
Research gate failure is a valid result and never blocks a completed model's
TEST. A technical failure is exported and isolated to its arm. Source or code
changes are fatal to the whole suite because they invalidate fair comparison.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from taxosafe_refine.importer import inspect_reference
from . import protocol

STAGES = ("training", "calibration", "test")
STAGE_MARKER = "stage_binding.json"


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Expected a regular file without symlink ancestors: " + str(path))
    return path


def _source(directory):
    directory = Path(directory).resolve()
    required = ("training/config.json", "training/inputs.json", "training/completed.json",
                "training/best.pth", "training/support.pth", "calibration/router.json",
                "calibration/completed.json", "calibration/development_scores.jsonl")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError("The trained reference is incomplete at {}. Missing: {}. "
                         "Use the intact server reference run; a text review archive cannot replace model weights."
                         .format(directory, ", ".join(missing)))
    return inspect_reference(directory)


def _destinations(reference_directory, suite_directory):
    reference = Path(reference_directory).resolve()
    suite = Path(suite_directory).resolve()
    if reference == suite or reference in suite.parents or suite in reference.parents:
        raise ValueError("Reference and sweep suite must be separate, non-nested directories")
    if any(p.is_symlink() for p in (Path(suite_directory), *Path(suite_directory).parents)):
        raise ValueError("Suite directory must not traverse symlinks")
    return reference, suite


def _device_check(device):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Run the suite in the server's working ProTeCt GPU environment; "
                         "--device cpu explicitly requests real CPU execution, not a simulated run.")
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device": device,
            "device_count": torch.cuda.device_count(),
            "gpu_memory_bytes": list(torch.cuda.mem_get_info()) if device == "cuda" and hasattr(torch.cuda, "mem_get_info") else None}


def preflight(cfg, reference_directory, suite_directory, device="cuda", resume=False):
    """Validate source artifacts, locked data and device without creating output."""
    reference_directory, suite = _destinations(reference_directory, suite_directory)
    source = _source(reference_directory)
    runtime = _device_check(device)
    from taxosafe_support import pipeline as support
    from taxosafe_refine.pipeline import _stage_rows
    groups, train_audit = support.load_stage_rows(source["config"], "train", source["meta"])
    for split, audit in train_audit.items():
        if audit != source["training"]["audit"][split]:
            raise ValueError("Locked training/selection data differ from the source: " + split)
    del groups
    audits = dict(train_audit)
    reference = SimpleNamespace(**source)
    for stage in ("calibrate", "test"):
        groups, audit = _stage_rows(reference, stage)
        audits.update(audit)
        del groups
    if resume:
        snapshot = _resume_audit(suite, cfg, source)
        if snapshot["device"] != device:
            raise ValueError("Device differs from the immutable suite snapshot")
    elif suite.exists():
        raise ValueError("Use a fresh absent suite directory; preserve existing runs: " + str(suite))
    return {"schema_version": protocol.SCHEMA_VERSION, "source_binding": source["binding"],
            "signature": protocol.signature(cfg, source["binding"]), "runtime": runtime,
            "locked_split_counts": {key: value["unique_image_count"] for key, value in audits.items()},
            "checkpoint_tensors_loaded": False, "model_forward_performed": False,
            "test_predictions_read": False, "destination_created": False}


def _initialize(suite, cfg, source, device):
    suite.mkdir(parents=True, exist_ok=False)
    signature = protocol.signature(cfg, source["binding"])
    protocol.write_json(suite / "config.json", cfg)
    protocol.write_json(suite / "source_binding.json", source["binding"])
    snapshot = {"schema_version": protocol.SCHEMA_VERSION, "created_at_utc": _stamp(),
                "config_sha256": protocol.object_hash(cfg), "signature": signature,
                "source_binding": source["binding"], "arm_ids": [a["id"] for a in cfg["arms"]],
                "device": device, "evaluation_order": "all_development_then_freeze_then_all_test",
                "test_on_failed_research_gate": True, "test_used_for_selection": False}
    protocol.write_json(suite / "snapshot.json", snapshot)
    return snapshot


def _verify_snapshot(suite, cfg, source):
    suite = Path(suite)
    snapshot = protocol.read_json(_regular(suite / "snapshot.json"))
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != protocol.object_hash(cfg)
            or snapshot.get("signature") != protocol.signature(cfg, source["binding"])
            or snapshot.get("source_binding") != source["binding"]
            or snapshot.get("arm_ids") != [a["id"] for a in cfg["arms"]]
            or protocol.read_json(_regular(suite / "config.json")) != cfg
            or protocol.read_json(_regular(suite / "source_binding.json")) != source["binding"]):
        raise ValueError("Suite source/configuration/code snapshot changed; do not bypass hashes or resume mixed versions")
    return snapshot


def _manifest(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Stage artifact symlink is prohibited: " + str(path))
        if path.is_file() and path != directory / STAGE_MARKER:
            _regular(path)
            result[path.relative_to(directory).as_posix()] = protocol.file_hash(path)
    if "completed.json" not in result:
        raise ValueError("Stage has no completed receipt: " + str(directory))
    return result


def _complete_stage(suite, arm_id, stage, snapshot):
    directory = Path(suite) / "arms" / arm_id / stage
    marker = directory / STAGE_MARKER
    if marker.exists():
        raise ValueError("Cannot overwrite a verified stage binding")
    protocol.write_json(marker, {"schema_version": protocol.SCHEMA_VERSION, "arm_id": arm_id,
        "stage": stage, "suite_signature": snapshot["signature"],
        "snapshot_sha256": protocol.file_hash(Path(suite) / "snapshot.json"),
        "files": _manifest(directory)})


def _verify_stage(suite, arm_id, stage, snapshot):
    directory = Path(suite) / "arms" / arm_id / stage
    marker = protocol.read_json(_regular(directory / STAGE_MARKER))
    if (marker.get("schema_version") != protocol.SCHEMA_VERSION or marker.get("arm_id") != arm_id
            or marker.get("stage") != stage or marker.get("suite_signature") != snapshot["signature"]
            or marker.get("snapshot_sha256") != protocol.file_hash(Path(suite) / "snapshot.json")
            or marker.get("files") != _manifest(directory)):
        raise ValueError("Completed stage artifacts changed: " + str(directory))
    return protocol.read_json(directory / "completed.json")


def _resume_audit(suite, cfg, source):
    snapshot = _verify_snapshot(suite, cfg, source)
    for arm in cfg["arms"]:
        arm_dir = Path(suite) / "arms" / arm["id"]
        if (arm_dir / "failure.json").exists():
            raise ValueError("Cannot resume a technically failed arm; keep its audit and choose a fresh suite: " + arm["id"])
        for stage in STAGES:
            directory = arm_dir / stage
            if directory.exists():
                if not (directory / STAGE_MARKER).is_file():
                    raise ValueError("Incomplete {} stage for {}; choose a fresh suite, do not delete receipts".format(stage, arm["id"]))
                _verify_stage(suite, arm["id"], stage, snapshot)
        if arm["kind"] == "finetune" and (arm_dir / "calibration").exists() and not (arm_dir / "training").exists():
            raise ValueError("Calibration is missing its training stage: " + arm["id"])
        weight_source = arm.get("weights")
        if (weight_source not in (None, "reference")
                and any((arm_dir / stage).exists() for stage in ("calibration", "test"))):
            dependency = suite / "arms" / weight_source / "training"
            if not dependency.is_dir():
                raise ValueError("Routing stage is missing its trained weight dependency: " + arm["id"])
            _verify_stage(suite, weight_source, "training", snapshot)
        if (arm_dir / "test").exists() and (not (arm_dir / "calibration").exists()
                or not (Path(suite) / "dev_selection.json").is_file()):
            raise ValueError("TEST is missing its calibrated model or frozen DEV selection: " + arm["id"])
    return snapshot


def _binding(suite, source, arm_id, checkpoint, support):
    snapshot = protocol.read_json(Path(suite) / "snapshot.json")
    return {"arm_id": arm_id, "source_binding": source.binding,
            "suite_signature": snapshot["signature"],
            "config_sha256": snapshot["config_sha256"],
            "suite_snapshot_sha256": protocol.file_hash(Path(suite) / "snapshot.json"),
            "checkpoint": {"path": str(Path(checkpoint).resolve()), "sha256": protocol.file_hash(checkpoint)},
            "support": {"path": str(Path(support).resolve()), "sha256": protocol.file_hash(support)}}


def worker(suite_directory, arm_id, stage, device):
    """One real isolated stage. No gate boolean controls the TEST branch."""
    import torch
    from taxosafe_refine.importer import load_reference
    from . import training, evaluation
    suite = Path(suite_directory).resolve()
    cfg = protocol.read_json(suite / "config.json")
    recorded = protocol.read_json(suite / "source_binding.json")
    info = _source(recorded["directory"])
    snapshot = _verify_snapshot(suite, cfg, info)
    if snapshot["device"] != device:
        raise ValueError("Device differs from the immutable suite snapshot")
    _device_check(device)
    arm = next((a for a in cfg["arms"] if a["id"] == arm_id), None)
    if arm is None or stage not in STAGES:
        raise ValueError("Unknown suite arm or stage")
    output = suite / "arms" / arm_id / stage
    if output.exists():
        raise ValueError("Stage output already exists; use verified resume from the suite runner")
    source = load_reference(info["directory"], torch.device(device))
    if stage == "training":
        if arm["kind"] != "finetune":
            raise ValueError("Only A01 performs neural training")
        training.train_arm(source, arm, copy.deepcopy(source.config), output,
                           torch.device(device), cfg["seed"], cfg["budget"])
    else:
        weight_source = arm["id"] if arm["kind"] == "finetune" else arm["weights"]
        if weight_source == "reference":
            encoder, evidence, bank = source.encoder, source.evidence, source.bank
            checkpoint, support = source.directory / "training/best.pth", source.directory / "training/support.pth"
        else:
            train_dir = suite / "arms" / weight_source / "training"
            _verify_stage(suite, weight_source, "training", snapshot)
            encoder, evidence, bank, receipt = training.load_arm_model(source, train_dir, torch.device(device))
            checkpoint, support = train_dir / receipt["checkpoint"]["path"], train_dir / receipt["support"]["path"]
        binding = _binding(suite, source, arm_id, checkpoint, support)
        variant = arm.get("router") if arm["kind"] == "routing" else None
        kwargs = dict(frozen_baseline=arm["kind"] == "baseline", route_variant=variant,
                      settings=cfg["router"], proximity_settings=cfg["proximity"] if variant else None)
        if stage == "calibration":
            reuse = suite / "arms/A03_combined/calibration"
            if arm_id == "A04_parent_rerank" and (reuse / STAGE_MARKER).is_file():
                _verify_stage(suite, "A03_combined", "calibration", snapshot)
                kwargs["reuse_proximity_dir"] = reuse
            evaluation.evaluate_development(source, encoder, evidence, bank, source.config, output,
                                            binding, **kwargs)
        else:
            _regular(suite / "dev_selection.json")
            _verify_stage(suite, arm_id, "calibration", snapshot)
            from .reporting import freeze_dev_selection
            freeze_dev_selection(suite)
            evaluation.evaluate_test(source, encoder, evidence, bank, source.config, output, binding,
                suite / "arms" / arm_id / "calibration", **kwargs)
    _verify_snapshot(suite, cfg, _source(recorded["directory"]))
    _complete_stage(suite, arm_id, stage, snapshot)


def _launch_stage(suite, arm_id, stage, device):
    logs = Path(suite) / "logs" / arm_id
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
    command = [sys.executable, "-u", "-m", "taxosafe_routealign", "--worker-stage", stage,
               "--arm", arm_id, "--suite-dir", str(suite), "--device", device]
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=str(protocol.PROJECT_ROOT), stdout=out, stderr=err, check=False)
    return result.returncode, stdout, stderr


def _failure(suite, arm_id, stage, message, stderr=None):
    directory = Path(suite) / "arms" / arm_id
    directory.mkdir(parents=True, exist_ok=True)
    record = {"arm_id": arm_id, "stage": stage, "error": message,
              "technical_failure": True, "research_gate_failure": False,
              "test_blocked": stage != "test", "created_at_utc": _stamp(),
              "stderr_log": None if stderr is None else str(Path(stderr).relative_to(suite))}
    protocol.write_json(directory / "failure.json", record)
    return record


def _execute(suite, cfg, source_directory, arm_id, stage, device, resume=False):
    source = _source(source_directory)
    snapshot = _verify_snapshot(suite, cfg, source)
    directory = Path(suite) / "arms" / arm_id / stage
    if directory.exists():
        if not resume:
            raise ValueError("Stage unexpectedly exists: " + str(directory))
        _verify_stage(suite, arm_id, stage, snapshot)
        return True
    print("[{}] {} {} started".format(_stamp(), arm_id, stage), flush=True)
    try:
        returncode, stdout, stderr = _launch_stage(suite, arm_id, stage, device)
    except OSError as exc:
        _verify_snapshot(suite, cfg, _source(source_directory))
        _failure(suite, arm_id, stage, "Cannot launch isolated stage: " + str(exc))
        return False
    # Unlike an arm error, changed provenance invalidates the entire suite.
    _verify_snapshot(suite, cfg, _source(source_directory))
    if returncode:
        message = stderr.read_text(encoding="utf-8", errors="replace")[-6000:]
        _failure(suite, arm_id, stage, "stage exit {}: {}".format(returncode, message), stderr)
        print("[{}] {} {} failed; remaining arms continue; see {}".format(_stamp(), arm_id, stage, stderr), flush=True)
        return False
    try:
        receipt = _verify_stage(suite, arm_id, stage, snapshot)
    except (ValueError, OSError) as exc:
        _failure(suite, arm_id, stage, "Stage did not produce a verified completion: " + str(exc), stderr)
        return False
    print("[{}] {} {} completed; targets_passed={}".format(_stamp(), arm_id, stage,
          receipt.get("targets_passed")), flush=True)
    return True


def execute_suite(cfg, reference_directory, suite_directory, device="cuda", resume=False, run_preflight=True):
    """No parameter or arm is selected using TEST; gate-failed arms still test."""
    from . import reporting
    reference_directory, suite = _destinations(reference_directory, suite_directory)
    if run_preflight:
        preflight(cfg, reference_directory, suite, device, resume)
    source = _source(reference_directory)
    if resume:
        snapshot = _resume_audit(suite, cfg, source)
    else:
        snapshot = _initialize(suite, cfg, source, device)
    if snapshot["device"] != device:
        raise ValueError("Device differs from the immutable suite snapshot")
    failed = set()
    with protocol.run_lock(suite):
        for arm in cfg["arms"]:
            arm_id = arm["id"]
            dependency = arm.get("weights")
            if dependency not in (None, "reference") and not (suite / "arms" / dependency / "training" / STAGE_MARKER).is_file():
                _failure(suite, arm_id, "dependency", "Required training is unavailable: " + dependency)
                failed.add(arm_id)
                continue
            if arm["kind"] == "finetune" and not _execute(suite, cfg, reference_directory, arm_id, "training", device, resume):
                failed.add(arm_id)
                continue
            if not _execute(suite, cfg, reference_directory, arm_id, "calibration", device, resume):
                failed.add(arm_id)
        _verify_snapshot(suite, cfg, _source(reference_directory))
        reporting.freeze_dev_selection(suite)
        frozen_hash = protocol.file_hash(_regular(suite / "dev_selection.json"))
        tested = []
        for arm in cfg["arms"]:
            if arm["id"] in failed:
                continue
            if protocol.file_hash(suite / "dev_selection.json") != frozen_hash:
                raise ValueError("Frozen DEV selection changed before TEST")
            if _execute(suite, cfg, reference_directory, arm["id"], "test", device, resume):
                tested.append(arm["id"])
            if protocol.file_hash(suite / "dev_selection.json") != frozen_hash:
                raise ValueError("TEST changed the frozen DEV selection")
        _verify_snapshot(suite, cfg, _source(reference_directory))
        summary = reporting.summarize_suite(suite, phase="complete")
        protocol.write_json(suite / "suite_completed.json", {
            "schema_version": protocol.SCHEMA_VERSION, "signature": snapshot["signature"],
            "workflow_completed": True, "dev_selection_sha256": frozen_hash,
            "all_valid_calibrations_test_attempted_regardless_of_gate": True,
            "completed_test_arms": tested,
            "technical_failure_arms": [a["id"] for a in cfg["arms"] if (suite / "arms" / a["id"] / "failure.json").exists()],
            "test_used_for_selection": False, "completed_at_utc": _stamp()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--reference-run-dir")
    parser.add_argument("--run-dir", "--suite-dir", dest="suite_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue only verified complete stages; partial/failed stages require a fresh suite")
    parser.add_argument("--worker-stage", choices=STAGES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_stage:
        if not args.arm or args.preflight or args.resume:
            parser.error("Worker requires an arm and cannot combine preflight/resume")
        worker(args.suite_dir, args.arm, args.worker_stage, args.device)
        return
    if not args.reference_run_dir:
        parser.error("--reference-run-dir is required; choose the intact trained reference explicitly")
    path = Path(args.config) if Path(args.config).is_file() else protocol.resolve(args.config)
    cfg = protocol.effective_config(path)
    reference = protocol.resolve(args.reference_run_dir)
    suite = protocol.resolve(str(args.suite_dir))
    if args.preflight:
        print(json.dumps(preflight(cfg, reference, suite, args.device, args.resume), indent=2))
        return
    try:
        execute_suite(cfg, reference, suite, args.device, args.resume)
    finally:
        if (suite / "snapshot.json").is_file():
            from tools.pack_taxosafe_routealign_review import pack
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archive = suite.with_name(suite.name + "_review_" + stamp + ".tar.gz")
            pack(suite, archive)
    print("RouteAlign complete: " + str(suite), flush=True)


if __name__ == "__main__":
    main()
