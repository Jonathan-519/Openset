"""Internal, read-only bridge from the exact historical Discovery contract.

This process imports only the archived namespace. It verifies the original
source, receipts, every declared artifact, cache alignment, geometry and BCE
normalization before exporting an explicit new interchange packet. It never
trains a model or edits historical files.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

LEGACY = Path(__file__).resolve().parent / "legacy"
sys.path.insert(0, str(LEGACY))
ARCHIVED_DISCOVERY_CODE = "ec445a6908d0e56d14060e4a2ad9561dcbcd9421fdf34877bc3de82f2059fe78"
EXPORT_SCHEMA = "h02_verified_d05_export_v1"


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("train", "test"), default="train")
    parser.add_argument("--frozen-development", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Interchange output must not exist")
    from taxosafe_discovery import protocol, runner, backend, calibration
    from taxosafe_recovery.importer import load_d05, load_parent_cache
    from taxosafe_support import pipeline as support
    from taxosafe_support import calibration as base
    if protocol.code_signature() != ARCHIVED_DISCOVERY_CODE:
        raise ValueError("Archived Discovery source differs from the reviewed original commit")
    if args.stage == "test":
        if args.frozen_development is None:
            parser.error("TEST export requires an already frozen TaxoSieve development receipt")
        frozen = protocol.read_json(runner._regular(args.frozen_development))
        if (frozen.get("schema_version") != "taxosieve_v1"
                or frozen.get("stage") != "calibration" or frozen.get("fit_completed") is not True
                or frozen.get("test_used_for_fitting") is not False):
            raise ValueError("Missing or invalid TaxoSieve development freeze")
    parent = load_d05(args.source)
    info = parent.info
    files = {str(info["directory"] / name): digest
             for name, digest in parent.binding["parent"]["artifacts"].items()}
    reference = info["reference"]
    files.update({str(reference["directory"] / name): digest
                  for name, digest in reference["binding"]["receipt_sha256"].items()})
    for key in ("checkpoint", "support", "router"):
        files[str(reference[key + "_path"])] = reference["binding"][key + "_sha256"]
    source_scores = runner._regular(reference["directory"] / "calibration/development_scores.jsonl")
    files[str(source_scores)] = protocol.file_hash(source_scores)
    packet = dict(schema_version=EXPORT_SCHEMA, stage=args.stage,
        archived_code_sha256=ARCHIVED_DISCOVERY_CODE,
        reference_directory=str(reference["directory"]), reference_config=reference["config"],
        parent_binding=copy.deepcopy(parent.binding), files=files,
        meta=parent.meta, config=info["config"], router=parent.router)
    if args.stage == "train":
        packet.update(payload=parent.payload, caches={}, audits={})
        for stage in ("train", "development"):
            cache, receipt = load_parent_cache(parent, stage)
            packet["caches"][stage] = cache
            packet["audits"][stage] = receipt["audit"]
        location = info["directory"] / "arms/D05_episode_bce/calibration"
        for name in ("scores", "predictions"):
            descriptor = info["calibration"]["artifacts"][name]
            path = runner._regular(location / descriptor["path"])
            if protocol.file_hash(path) != descriptor["sha256"]:
                raise ValueError("Original D05 development artifact changed")
            files[str(path)] = descriptor["sha256"]
            packet["development_" + name] = base.unique_records(read_records(path))
        # Check the actual saved terminals as well as the original reporting
        # audit done by load_d05. This packet contains DEV only.
        reproduced = calibration.decode_records(packet["development_scores"], parent.router, parent.meta)
        fields = ("candidate_leaf", "candidate_parent", "prediction_type", "leaf", "parent", "output_node")
        before = {row["image_sha256"]: row for row in packet["development_predictions"]}
        if any(any(row.get(k) != before[row["image_sha256"]].get(k) for k in fields) for row in reproduced):
            raise ValueError("Original D05 development terminals fail reproduction")
    else:
        # The new caller has already validated its own complete DEV artifacts;
        # the original TEST representation must also belong to its own freeze.
        runner._verify_stage(info["directory"], None, "cache_test", info["snapshot"])
        cache, receipt = backend._load_cache(info["directory"], "test", info["config"], reference)
        original_freeze = runner._regular(info["directory"] / "dev_selection.json")
        if (receipt.get("dev_selection_sha256") != protocol.file_hash(original_freeze)
                or receipt.get("inference_spec_sha256") != info["training"]["inference_spec_sha256"]):
            raise ValueError("Original TEST cache is not bound to the original frozen DEV and representation")
        packet.update(cache=cache, audit=receipt["audit"],
                      frozen_development_sha256=protocol.file_hash(args.frozen_development))
        packet["test_source_files"] = {str(original_freeze): protocol.file_hash(original_freeze)}
        for path in (info["directory"] / "cache/test").iterdir():
            packet["test_source_files"][str(runner._regular(path))] = protocol.file_hash(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    support._save_torch(args.output, packet)
    print("Verified historical D05 {} export completed.".format(args.stage))


if __name__ == "__main__":
    main()
