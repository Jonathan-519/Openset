"""The complete TaxoSieve lifecycle, with one model and no experiment matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from . import protocol as p


def _torch_load(path):
    from taxosafe_support.pipeline import _load_torch
    return _load_torch(p.regular(path))


def _torch_save(path, value):
    from taxosafe_support.pipeline import _save_torch
    _save_torch(path, value)


def _require_image_device(device):
    import torch
    # The inherited MaPLe implementation constructs CUDA prompt tensors. CPU is
    # supported for D05 cache training, score replay, calibration and decoding.
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reference training/image extraction requires CUDA. "
                           "Use a CUDA host, or import the original D05 feature caches for CPU calibration.")
    return torch.device(device)


def _flatten(groups):
    return [row for rows in groups.values() for row in rows]


def _by_status(groups):
    from taxosafe_support.calibration import STATUSES
    rows = _flatten(groups)
    return [[row for row in rows if row["status"] == status] for status in STATUSES]


def _semantic_digest(rows, fields):
    from taxosafe_support.calibration import unique_records
    unique = sorted(unique_records(rows), key=lambda r: r["image_sha256"])
    return p.object_hash([dict(image_sha256=r["image_sha256"],
                              **{key: r.get(key) for key in fields}) for r in unique])


SCORE_FIELDS = ("discovery", "support_evidence", "log_probs", "global_pred_leaf")
TERMINAL_FIELDS = ("candidate_leaf", "candidate_parent", "prediction_type", "leaf", "parent", "output_node")


def _reference_recipe(run):
    from taxosafe_support.protocol import effective_config
    return effective_config(p.resolve(run["config"]["reference_config"]), seed=run["config"]["d05"]["seed"])


def _prepare_reference(directory, run):
    from taxosafe_support import pipeline as support
    from .source import inspect_reference
    reference = Path(run["reference_directory"])
    if reference == Path(directory).resolve() / "reference":
        cfg = _reference_recipe(run)
        if not (reference / "training/completed.json").is_file():
            device = _require_image_device(run["device"])
            support.seed_all(cfg["seed"])
            with p.run_lock(reference):
                support.train(cfg, reference, device, debug=False)
        if not (reference / "calibration/completed.json").is_file():
            device = _require_image_device(run["device"])
            support.seed_all(cfg["seed"])
            with p.run_lock(reference):
                support.calibrate_run(cfg, reference, device)
    info = inspect_reference(reference)
    if p.object_hash(info["config"]) != run["reference_config_sha256"]:
        raise ValueError("The reference is not the locked TaxoSieve reference recipe")
    p.save_source(directory, info["binding"])
    return info


def _validate_cache(cache, audit, stage, info, source):
    from . import d05
    from .source import _validate_audit
    d05.validate_cache(cache, info["meta"], stage)
    _validate_audit(audit, cache["groups"], allow_duplicates=stage == "test")
    allowed_origins = [info["binding"]]
    if source.get("legacy"):
        allowed_origins.append(source["legacy"]["parent_binding"]["reference_binding"])
    if (cache["provenance"].get("source_binding") not in allowed_origins
            or cache["provenance"].get("preprocessing") != info["config"]["data"]):
        raise ValueError("Feature cache does not belong to the verified reference and preprocessing")
    for split, group in cache["groups"].items():
        if (set(group["image_sha256"]) != set(audit[split]["image_hashes"])
                or len(group["records"]) != audit[split]["count"]
                or (split in info["audit"] and audit[split] != info["audit"][split])):
            raise ValueError("Cached identities/counts differ from the frozen split audit: " + split)
    return cache


def _store_cache(directory, stage, cache, audit, info, source, **details):
    from .d05 import _text_contract
    _validate_cache(cache, audit, stage, info, source)
    output = p.claim_stage(directory, "cache/" + stage)
    _torch_save(output / "features.pth", cache)
    receipt = p.finish_stage(directory, "cache/" + stage,
        dict(audit=audit, meta=info["meta"], inference_spec_sha256=_text_contract(cache), **details),
        dict(features="features.pth"))
    return cache, receipt


def _load_cache(directory, stage, info, source):
    from .d05 import _text_contract
    receipt = p.verify_stage(directory, "cache/" + stage)
    cache = _torch_load(p.artifact(p.stage_path(directory, "cache/" + stage), receipt["artifacts"]["features"]))
    _validate_cache(cache, receipt["audit"], stage, info, source)
    if receipt.get("meta") != info["meta"] or _text_contract(cache) != receipt.get("inference_spec_sha256"):
        raise ValueError("Cached inference contract or taxonomy changed")
    return cache, receipt


def _collect_cache(directory, stage, info, source, run):
    from . import d05
    from .source import load_reference, load_training_rows, stage_rows, check_baseline_development
    if p.completed(directory, "cache/" + stage):
        return _load_cache(directory, stage, info, source)
    if stage == "test":
        p.verify_stage(directory, "calibration")
    device = _require_image_device(run["device"])
    reference = load_reference(info["directory"], device)
    if reference.binding != info["binding"]:
        raise ValueError("Reference changed before image extraction")
    if stage == "train":
        rows, audit = load_training_rows(reference)
        groups, audit = {"train": rows}, {"train": audit}
    else:
        groups, audit = stage_rows(reference, "calibrate" if stage == "development" else "test")
    cache = d05.collect_cache(reference, groups, device)
    details = {}
    if stage == "development":
        details["reference_reproduction"] = check_baseline_development(reference,
            {key: group["records"] for key, group in cache["groups"].items()})
    if stage == "test":
        details["frozen_development_sha256"] = p.file_hash(Path(directory) / "calibration/completed.json")
    return _store_cache(directory, stage, cache, audit, info, source, **details)


def _load_training(directory, info, source):
    from . import d05
    cache, cached = _load_cache(directory, "train", info, source)
    trained = p.verify_stage(directory, "training")
    payload = _torch_load(p.artifact(p.stage_path(directory, "training"), trained["artifacts"]["model"]))
    d05.validate_payload(payload, info["meta"], cache)
    if (trained.get("train_cache_sha256") != cached["artifacts"]["features"]["sha256"]
            or trained.get("inference_spec_sha256") != cached["inference_spec_sha256"]
            or trained.get("fit_report") != payload["fit_report"]
            or trained.get("gradient_splits") != ["train"]):
        raise ValueError("D05 training provenance changed")
    return payload, trained


def _verify_calibration(directory, info, source, trained):
    from . import calibration, d05_calibration
    calibrated = p.verify_stage(directory, "calibration")
    _, cached = _load_cache(directory, "development", info, source)
    if (calibrated.get("fit_completed") is not True
            or calibrated.get("fit_splits") != ["val_known", "val_intra", "val_extra"]
            or calibrated.get("meta") != info["meta"]
            or calibrated.get("model_sha256") != trained["artifacts"]["model"]["sha256"]
            or calibrated.get("training_receipt_sha256") != p.file_hash(Path(directory) / "training/completed.json")
            or calibrated.get("development_cache_sha256") != cached["artifacts"]["features"]["sha256"]
            or cached.get("inference_spec_sha256") != trained["inference_spec_sha256"]):
        raise ValueError("Calibration model/cache/fitting provenance changed")
    router = p.read_json(p.artifact(Path(directory) / "calibration", calibrated["artifacts"]["router"]))
    drouter = p.read_json(p.artifact(Path(directory) / "calibration", calibrated["artifacts"]["d05_router"]))
    calibration.validate_router(router, info["meta"])
    d05_calibration.validate_router(drouter, info["meta"])
    return calibrated


def train(directory, reference_directory=None, device="cuda", config=p.DEFAULT_CONFIG, resume=False):
    from . import d05
    directory = Path(directory).resolve()
    if directory.exists():
        if not resume:
            raise ValueError("Run directory already exists; use --resume to verify and continue completed stages")
        run = p.inspect_run(directory)
        if run["mode"] != "train" or run["device"] != device:
            raise ValueError("Training mode/device differs from this run")
        if reference_directory is not None and str(Path(reference_directory).resolve()) != run["reference_directory"]:
            raise ValueError("Reference source differs from this run")
        if p.effective_config(config) != run["config"]:
            raise ValueError("Configuration differs from this run")
    else:
        cfg = p.effective_config(config)
        reference_directory = directory / "reference" if reference_directory is None else p.resolve(reference_directory).resolve()
        # External sources must be compatible before reserving a new run. The
        # native reference below does not exist until its own training stage.
        if reference_directory != directory / "reference":
            from .source import inspect_reference
            info = inspect_reference(reference_directory)
            expected_reference = _reference_recipe({"config": cfg})
            if p.object_hash(info["config"]) != p.object_hash(expected_reference):
                raise ValueError("The reference is not the locked TaxoSieve reference recipe")
        run = p.initialize(directory, cfg, reference_directory, device)
    with p.run_lock(directory):
        _prepare_reference(directory, run)
        info, source = p.inspect_source(directory)
        if p.completed(directory, "training"):
            return _load_training(directory, info, source)[1]
        cache, cached = _collect_cache(directory, "train", info, source, run)
        output = p.claim_stage(directory, "training")
        payload, report = d05.fit_payload(cache, **run["config"]["d05"])
        d05.validate_payload(payload, info["meta"], cache)
        _torch_save(output / "model.pth", payload)
        p.inspect_source(directory)
        return p.finish_stage(directory, "training",
            dict(meta=info["meta"], fit_report=report, optimizer_steps=report["optimizer_steps"],
                 optimizer_steps_in_this_run=report["optimizer_steps"], gradient_splits=["train"],
                 training_execution="D05_BCE_known_TRAIN", train_cache_sha256=cached["artifacts"]["features"]["sha256"],
                 inference_spec_sha256=cached["inference_spec_sha256"]), dict(model="model.pth"))


def _historical_export(discovery_directory, output, stage="train", frozen=None):
    command = [sys.executable, str(p.PROJECT_ROOT / "comparison_experiments/_export_d05.py"),
               "--source", str(Path(discovery_directory).resolve()), "--output", str(output), "--stage", stage]
    if frozen is not None:
        command.extend(("--frozen-development", str(frozen)))
    subprocess.run(command, cwd=str(p.PROJECT_ROOT), check=True)
    packet = _torch_load(output)
    if packet.get("schema_version") != "h02_verified_d05_export_v1" or packet.get("stage") != stage:
        raise ValueError("Invalid historical interchange packet")
    if packet.get("archived_code_sha256") != "ec445a6908d0e56d14060e4a2ad9561dcbcd9421fdf34877bc3de82f2059fe78":
        raise ValueError("Historical interchange used an unreviewed runtime")
    for path, digest in packet["files"].items():
        if p.file_hash(p.regular(path)) != digest:
            raise ValueError("Historical artifact changed during export: " + path)
    return packet


def import_d05(discovery_directory, directory, device="cuda", config=p.DEFAULT_CONFIG):
    from . import d05, d05_calibration
    from .source import inspect_reference
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("D05 import requires a fresh run directory")
    cfg = p.effective_config(config)
    with tempfile.TemporaryDirectory(prefix="taxosieve_verified_d05_") as temporary:
        packet = _historical_export(discovery_directory, Path(temporary) / "d05.pth")
        info = inspect_reference(packet["reference_directory"])
        if packet["meta"] != info["meta"] or packet["reference_config"] != info["config"]:
            raise ValueError("D05/reference taxonomy or recipe differs")
        expected_reference = _reference_recipe({"config": cfg})
        if p.object_hash(info["config"]) != p.object_hash(expected_reference):
            raise ValueError("Imported source uses another reference recipe")
        old = packet["config"]
        expected = dict(seed=old["seed"], shrinkage=old["geometry"]["shrinkage"], hidden=32,
                        **{key: old["verifier"][key] for key in ("folds", "epochs", "batch_size", "lr")})
        if expected != cfg["d05"] or old["calibration"] != cfg["d05_calibration"]:
            raise ValueError("Imported D05 does not use the locked TaxoSieve training/calibration budget")
        d05.validate_payload(packet["payload"], info["meta"], packet["caches"]["train"])
        d05_calibration.validate_router(packet["router"], info["meta"])
        run = p.initialize(directory, cfg, info["directory"], device, mode="legacy_d05_import")
        if p.object_hash(info["config"]) != run["reference_config_sha256"]:
            raise ValueError("Imported source uses another reference recipe")
        legacy = dict(directory=str(Path(discovery_directory).resolve()), parent_binding=packet["parent_binding"],
            files=packet["files"], archived_code_sha256=packet["archived_code_sha256"],
            development_scores_sha256=_semantic_digest(packet["development_scores"], SCORE_FIELDS),
            development_terminals_sha256=_semantic_digest(packet["development_predictions"], TERMINAL_FIELDS))
        source = p.save_source(directory, info["binding"], legacy)
        with p.run_lock(directory):
            caches = {stage: _store_cache(directory, stage, packet["caches"][stage],
                packet["audits"][stage], info, source)[1] for stage in ("train", "development")}
            output = p.claim_stage(directory, "training")
            _torch_save(output / "model.pth", packet["payload"])
            p.write_json(output / "original_d05_router.json", packet["router"])
            report = packet["payload"]["fit_report"]
            p.inspect_source(directory)
            return p.finish_stage(directory, "training",
                dict(meta=info["meta"], fit_report=report, optimizer_steps=report["optimizer_steps"],
                    optimizer_steps_in_this_run=0, gradient_splits=["train"], training_execution="verified_original_D05_import",
                    train_cache_sha256=caches["train"]["artifacts"]["features"]["sha256"],
                    inference_spec_sha256=caches["train"]["inference_spec_sha256"]),
                dict(model="model.pth", original_d05_router="original_d05_router.json"))


def _predictions(groups, router, meta):
    from .calibration import decode_records
    return _flatten({split: decode_records(rows, router, meta) for split, rows in groups.items()})


def _write_predictions(path, rows):
    # Each input alias remains traceable; the metrics use unique content hashes.
    # Raw model evidence already lives in the cache and is not duplicated here.
    keep = ("image_sha256", "path", "image", "split", "status", "source", "true_parent", "true_leaf",
            *TERMINAL_FIELDS, "root_pass", "selected_root_score", "selected_leaf_score",
            "selected_parent_score", "leaf_margin", "root_margin", "root_threshold", "leaf_threshold")
    seen = set()
    records = []
    for row in rows:
        result = {key: row[key] for key in keep if key in row}
        result["evaluation_weight"] = int(row["image_sha256"] not in seen)
        seen.add(row["image_sha256"])
        records.append(result)
    p.write_records(path, records)


def calibrate(directory, save_scores=False):
    from . import calibration, d05_calibration, d05
    from .source import check_baseline_development
    from taxosafe_support import calibration as base
    directory = Path(directory).resolve()
    with p.run_lock(directory):
        run = p.inspect_run(directory)
        info, source = p.inspect_source(directory)
        payload, trained = _load_training(directory, info, source)
        if p.completed(directory, "calibration"):
            return _verify_calibration(directory, info, source, trained)
        cache, cached = _collect_cache(directory, "development", info, source, run)
        if cached["inference_spec_sha256"] != trained["inference_spec_sha256"]:
            raise ValueError("TRAIN and DEV inference representations differ")
        raw = {key: group["records"] for key, group in cache["groups"].items()}
        reproduction = check_baseline_development(SimpleNamespace(**info), raw)
        dgroups = d05.score_groups(cache, payload, info["meta"])
        reference_bundle = dict(records=_flatten(raw), router=info["router"],
                                calibration_settings=info["config"]["calibration"])
        legacy = source.get("legacy")
        if legacy is not None:
            drouter = p.read_json(p.artifact(directory / "training", trained["artifacts"]["original_d05_router"]))
            if _semantic_digest(_flatten(dgroups), SCORE_FIELDS) != legacy["development_scores_sha256"]:
                raise ValueError("Original D05 DEV evidence no longer reproduces exactly")
            terminals = d05_calibration.decode_records(_flatten(dgroups), drouter, info["meta"])
            if _semantic_digest(terminals, TERMINAL_FIELDS) != legacy["development_terminals_sha256"]:
                raise ValueError("Original D05 DEV terminal predictions changed")
        else:
            drouter, _ = d05_calibration.fit_router(*_by_status(dgroups), info["meta"],
                run["config"]["d05_calibration"], "global", reference_bundle)
        groups = calibration.augment_scores(dgroups, info["meta"])
        dbundle = dict(records=_flatten(dgroups), router=drouter)
        router, diagnostics = calibration.fit_router(*_by_status(groups), info["meta"],
            run["config"]["calibration"], "staged", reference_bundle, dbundle)
        crossfit = calibration.crossfit_audit(*_by_status(groups), info["meta"],
            run["config"]["calibration"], "staged", reference_bundle, dbundle)
        predictions = _predictions(groups, router, info["meta"])
        summary = base.evaluate_records(predictions, info["meta"])
        output = p.claim_stage(directory, "calibration")
        p.write_json(output / "router.json", router)
        p.write_json(output / "d05_router.json", drouter)
        p.write_json(output / "audit.json", dict(reference_reproduction=reproduction,
            staged_calibration=diagnostics, crossfit=crossfit))
        names = dict(router="router.json", d05_router="d05_router.json", audit="audit.json")
        if save_scores:
            p.write_records(output / "scores.jsonl", _flatten(groups))
            names["scores"] = "scores.jsonl"
        p.inspect_source(directory)
        return p.finish_stage(directory, "calibration",
            dict(meta=info["meta"], fit_completed=True, fit_splits=["val_known", "val_intra", "val_extra"],
                model_sha256=trained["artifacts"]["model"]["sha256"],
                training_receipt_sha256=p.file_hash(directory / "training/completed.json"),
                development_cache_sha256=cached["artifacts"]["features"]["sha256"], summary=summary,
                crossfit_complete=crossfit.get("complete"), crossfit_passed=crossfit.get("passed"),
                test_allowed_after_failed_research_gates=True), names)


def _source_test_cache(directory, info, source, frozen):
    if not source.get("legacy"):
        raise ValueError("--source-cache requires a run imported from the original D05")
    with tempfile.TemporaryDirectory(prefix="taxosieve_verified_test_") as temporary:
        packet = _historical_export(source["legacy"]["directory"], Path(temporary) / "test.pth",
                                    stage="test", frozen=directory / "calibration/completed.json")
        if (packet["parent_binding"] != source["legacy"]["parent_binding"]
                or packet["frozen_development_sha256"] != frozen):
            raise ValueError("Original TEST cache belongs to another D05 source or DEV freeze")
        return _store_cache(directory, "test", packet["cache"], packet["audit"], info, source,
            frozen_development_sha256=frozen, historical_test_files=packet["test_source_files"])


def test(directory, source_cache=False, save_scores=False):
    from . import calibration, d05
    from taxosafe_support import calibration as base
    directory = Path(directory).resolve()
    with p.run_lock(directory):
        run = p.inspect_run(directory)
        info, source = p.inspect_source(directory)
        # Check the DEV gate before attempting any TEST cache/image access.
        p.verify_stage(directory, "calibration")
        payload, trained = _load_training(directory, info, source)
        calibrated = _verify_calibration(directory, info, source, trained)
        frozen = p.file_hash(directory / "calibration/completed.json")
        if p.completed(directory, "test"):
            receipt = p.verify_stage(directory, "test")
            _, cached = _load_cache(directory, "test", info, source)
            if (receipt.get("frozen_development_sha256") != frozen
                    or cached.get("frozen_development_sha256") != frozen
                    or receipt.get("model_sha256") != trained["artifacts"]["model"]["sha256"]
                    or receipt.get("test_cache_sha256") != cached["artifacts"]["features"]["sha256"]
                    or cached.get("inference_spec_sha256") != trained["inference_spec_sha256"]):
                raise ValueError("Completed TEST model/cache/DEV binding changed")
            return receipt
        if source_cache and not p.completed(directory, "cache/test"):
            cache, cached = _source_test_cache(directory, info, source, frozen)
        else:
            cache, cached = _collect_cache(directory, "test", info, source, run)
        if (cached["inference_spec_sha256"] != trained["inference_spec_sha256"]
                or cached.get("frozen_development_sha256") != frozen):
            raise ValueError("TEST representation or development binding changed")
        groups = calibration.augment_scores(d05.score_groups(cache, payload, info["meta"]), info["meta"])
        router = p.read_json(p.artifact(directory / "calibration", calibrated["artifacts"]["router"]))
        predictions = _predictions(groups, router, info["meta"])
        summary = base.evaluate_records(predictions, info["meta"])
        if p.file_hash(directory / "calibration/completed.json") != frozen:
            raise ValueError("DEV freeze changed during TEST")
        p.inspect_source(directory)
        p.verify_stage(directory, "calibration")
        output = p.claim_stage(directory, "test")
        _write_predictions(output / "predictions.jsonl", predictions)
        names = dict(predictions="predictions.jsonl")
        if save_scores:
            p.write_records(output / "scores.jsonl", _flatten(groups))
            names["scores"] = "scores.jsonl"
        return p.finish_stage(directory, "test",
            dict(meta=info["meta"], frozen_development_sha256=frozen,
                model_sha256=trained["artifacts"]["model"]["sha256"],
                test_cache_sha256=cached["artifacts"]["features"]["sha256"],
                summary=summary, metric_unit="unique_image_sha256", statistics_refitted=False), names)


def preflight(config=p.DEFAULT_CONFIG, reference_directory=None, metadata_only=False):
    from taxosafe_support import pipeline as support
    from taxosafe_support import protocol as sp
    import torch
    cfg = p.effective_config(config)
    ref = sp.effective_config(p.resolve(cfg["reference_config"]), seed=cfg["d05"]["seed"])
    meta = support.hierarchy(ref)
    splits = {split for values in sp.STAGE_SPLITS.values() for split in values}
    manifests = {split: dict(path=ref["data"][split], sha256=p.file_hash(p.resolve(ref["data"][split])),
        record_count=sum(bool(line.strip()) for line in p.resolve(ref["data"][split]).read_text(encoding="utf-8").splitlines()))
        for split in sorted(splits)}
    result = dict(version=cfg["version"], python=sys.version.split()[0], torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
        reference_signature=sp.signature(ref), meta=meta, manifests=manifests,
        image_bytes_checked=not metadata_only, model_forward_performed=False,
        checkpoints_present_in_git=False)
    if not metadata_only:
        _, training = sp.load_stage_rows(ref, "train", meta)
        forbidden = set(training["train"]["image_hashes"])
        _, development = sp.load_stage_rows(ref, "calibrate", meta, forbidden_hashes=forbidden)
        sources = set()
        for split, audit in development.items():
            forbidden.update(audit["image_hashes"])
            if split != "val_known":
                sources.update(audit["sources"])
        _, tested = sp.load_stage_rows(ref, "test", meta, forbidden_hashes=forbidden, forbidden_sources=sources)
        audits = dict(training, **development, **tested)
        result["unique_images"] = {split: audit["unique_image_count"] for split, audit in audits.items()}
    if reference_directory is not None:
        from .source import inspect_reference
        result["source_binding"] = inspect_reference(p.resolve(reference_directory))["binding"]
    return result


def replay(scores, router_path):
    from .source import read_records
    from .calibration import decode_records
    from taxosafe_support.calibration import evaluate_records
    router = p.read_json(router_path)
    records = read_records(scores)
    return evaluate_records(decode_records(records, router, router["meta"]), router["meta"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "all"):
        child = sub.add_parser(command, help="Train reference/D05" if command == "train" else "Run TRAIN, DEV calibration and frozen TEST")
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--reference-run-dir", type=Path)
        child.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        child.add_argument("--config", default=p.DEFAULT_CONFIG)
        child.add_argument("--resume", action="store_true")
        if command == "all":
            child.add_argument("--save-scores", action="store_true")
    child = sub.add_parser("import-d05", help="Strictly import the original D05 model and TRAIN/DEV caches")
    child.add_argument("--discovery-run-dir", type=Path, required=True)
    child.add_argument("--run-dir", type=Path, required=True)
    child.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="Device for any later image extraction")
    child.add_argument("--config", default=p.DEFAULT_CONFIG)
    for command in ("calibrate", "test"):
        child = sub.add_parser(command)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--save-scores", action="store_true", help="Optionally save full raw scores for detailed replay")
        if command == "test":
            child.add_argument("--source-cache", action="store_true", help="Reuse original TEST features, after new DEV is frozen")
    child = sub.add_parser("preflight", help="Read-only recipe, taxonomy, runtime and split validation")
    child.add_argument("--config", default=p.DEFAULT_CONFIG)
    child.add_argument("--reference-run-dir", type=Path)
    child.add_argument("--metadata-only", action="store_true", help="Validate locked metadata without opening image bytes")
    child = sub.add_parser("inspect", help="Verify an existing TaxoSieve run and its completed artifacts")
    child.add_argument("--run-dir", type=Path, required=True)
    child = sub.add_parser("replay", help="Decode precomputed TaxoSieve scores with a frozen router")
    child.add_argument("--scores", type=Path, required=True)
    child.add_argument("--router", type=Path, required=True,
                       help="Frozen router from the same run/data version as --scores")
    args = parser.parse_args()
    if args.command in ("train", "all"):
        result = train(args.run_dir, args.reference_run_dir, args.device, args.config, args.resume)
        if args.command == "all":
            calibrate(args.run_dir, args.save_scores)
            result = test(args.run_dir, save_scores=args.save_scores)
    elif args.command == "import-d05":
        result = import_d05(args.discovery_run_dir, args.run_dir, args.device, args.config)
    elif args.command == "calibrate":
        result = calibrate(args.run_dir, args.save_scores)
    elif args.command == "test":
        result = test(args.run_dir, args.source_cache, args.save_scores)
    elif args.command == "preflight":
        result = preflight(args.config, args.reference_run_dir, args.metadata_only)
    elif args.command == "replay":
        result = replay(args.scores, args.router)
    else:
        p.inspect_source(args.run_dir)
        result = {stage: p.verify_stage(args.run_dir, stage) for stage in
                  ("cache/train", "training", "cache/development", "calibration", "cache/test", "test")
                  if p.completed(args.run_dir, stage)}
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
