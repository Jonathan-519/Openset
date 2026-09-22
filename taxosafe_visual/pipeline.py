"""Three separate commands: train-memory, validation-calibration, final test."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import core, residual
from .runtime import (PROJECT_ROOT, assert_disjoint, extract, extraction_signature,
                      load_bank, load_configuration, load_model, read_json,
                      resolve, sha256, write_json, write_records)


def arguments(stage):
    p = argparse.ArgumentParser(description="TaxoSafe visual support: " + stage)
    p.add_argument("--config", required=True, help="configs/.../TaxoSafe_visual.yml")
    p.add_argument("--trial", default="4")
    p.add_argument("--run-dir", default=None, help="Existing trained run containing ckpt/best.pth")
    p.add_argument("--artifact-dir", default=None)
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace this stage's outputs")
    if stage == "test":
        p.add_argument("--profiles", choices=["all", "both", "balanced", "risk", "coverage"],
                       default="both", help="Predeclare profiles; all/both share ONE extraction pass")
    return p.parse_args()


def _setup(args):
    os.chdir(PROJECT_ROOT)
    cfg, run, checkpoint = load_configuration(args.config, args.trial, args.run_dir)
    default_folder = "residual_support_v5" if _residual_enabled(cfg["visual_support"]) else "visual_support_v4"
    folder = resolve(args.artifact_dir) if args.artifact_dir else run / default_folder
    folder.mkdir(parents=True, exist_ok=True)
    return cfg, run, checkpoint, folder


def _protect(paths, overwrite):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist: {}. Keep old results or explicitly use --overwrite.".format(existing))


def _settings_fingerprint(settings):
    import hashlib
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def _residual_enabled(settings):
    return bool(settings.get("residual", {}).get("enabled", False))


def _balanced_enabled(settings):
    return (_residual_enabled(settings)
            and bool(settings.get("residual", {}).get("balanced_profile", False)))


def _profile_pairs(settings, requested="both"):
    """Return (child profile, frozen root profile) pairs."""
    available = [("coverage", "coverage")]
    if _balanced_enabled(settings):
        available.append(("balanced", "coverage"))
    available.append(("risk", "risk"))
    if requested == "all":
        if not _balanced_enabled(settings):
            raise ValueError("The all profile request requires residual.balanced_profile=true")
        return available
    if requested == "both":
        return [pair for pair in available if pair[0] in ("coverage", "risk")]
    selected = [pair for pair in available if pair[0] == requested]
    if not selected:
        raise ValueError("Requested profile is not enabled: " + requested)
    return selected


def _calibration_schema(settings):
    if not _residual_enabled(settings):
        return 4
    if _balanced_enabled(settings):
        return 7
    mode = str(settings.get("residual", {}).get("threshold_mode", "branch_min"))
    return 5 if mode == "branch_min" else 6


def _implementation_signature():
    # v5 covers sources omitted by the original v4 extraction signature too.
    names = ["taxosafe_visual/" + n + ".py" for n in ("core", "residual", "pipeline", "runtime", "metrics")]
    names += ["models/maple_model.py", "models/simple_tokenizer.py", "loader/hierdata.py", "loader/utils.py"]
    bpe = PROJECT_ROOT / "models/bpe_simple_vocab_16e6.txt.gz"
    values = {name: sha256(PROJECT_ROOT / name) for name in names}
    values["bpe_sha256"] = sha256(bpe) if bpe.is_file() else "missing"
    return _settings_fingerprint(values)


def _load_residual(folder, info, settings):
    value = read_json(folder / "residual_state.json")
    if (value["schema_version"] != 1 or value["memory_sha256"] != info["memory_sha256"]
            or value["settings_fingerprint"] != _settings_fingerprint(settings)
            or value["implementation_signature"] != _implementation_signature()):
        raise ValueError("Residual state provenance changed; rebuild in a new artifact directory")
    return value["state"]


def _refine(baseline, out, calibration, meta, profile, method="parent_residual_local_support"):
    result = residual.apply(baseline, out, calibration, meta, profile, method)
    for a, b in zip(baseline, result):
        for key in ("root_knownness_score", "root_gate_margin", "candidate_parent", "parent"):
            if a[key] != b[key]:
                raise AssertionError("Residual method changed the frozen root: " + key)
        if (a["prediction_type"] == "global_unknown") != (b["prediction_type"] == "global_unknown"):
            raise AssertionError("Residual method changed global rejection")
    return result


def _baseline_evidence(predictions):
    return {"score": np.asarray([r["child_knownness_score"] for r in predictions]),
            "leaf": np.asarray([r["candidate_leaf"] for r in predictions]),
            "parent": np.asarray([r["candidate_parent"] for r in predictions]),
            "neighbour": np.asarray([r["support_neighbor_index"] for r in predictions])}


def _source_constraints(predictions, indices, settings):
    report = {}
    for status, limit in (("extra", settings.get("root_far_limit", .05)),
                          ("intra", settings.get("child_oser_limit", .05))):
        records = [predictions[i] for i in indices if predictions[i]["status"] == status]
        rates = {}
        for source in sorted({r["source"] for r in records}):
            group = [r for r in records if r["source"] == source]
            bad = [(r["prediction_type"] != "global_unknown" if status == "extra" else r["prediction_type"] == "known") for r in group]
            rates[source] = {"count": len(group), "rate": float(np.mean(bad)), "limit": float(limit)}
        report[status] = rates
    report["empirical_constraints_satisfied"] = all(v["rate"] <= v["limit"] + 1e-12 for s in ("extra", "intra") for v in report[s].values())
    return report


def build_memory():
    args = arguments("memory")
    cfg, run, checkpoint, folder = _setup(args)
    outputs = [folder / "memory.npz", folder / "memory.json"]
    if _residual_enabled(cfg["visual_support"]):
        outputs.append(folder / "residual_state.json")
    _protect(outputs, args.overwrite)
    model, texts, meta, device, scale = load_model(cfg, checkpoint)
    # This command reads train only (besides checkpoint, config and tree).
    rows, f, _, _ = extract(cfg, "train", model, texts, meta, device)
    by_hash, unique = {}, []
    for i, r in enumerate(rows):
        h, label = r["image_sha256"], r["true_leaf"]
        if h in by_hash and by_hash[h] != label:
            raise ValueError("Identical training image has contradictory leaf labels")
        if h not in by_hash:
            unique.append(i)
            by_hash[h] = label
    settings = cfg["visual_support"]
    cap = int(settings.get("max_memory_per_leaf", 256))
    if cap < 2:
        raise ValueError("max_memory_per_leaf must be >=2")
    rng = np.random.RandomState(int(settings.get("seed", 41)))
    keep = []
    for c in range(len(meta["leaf_names"])):
        indices = [i for i in unique if rows[i]["true_leaf"] == c]
        if not indices:
            raise ValueError("No memory examples for leaf {}".format(c))
        if len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False).tolist()
        keep.extend(sorted(indices))
    keep = np.asarray(sorted(keep), dtype=np.int64)
    labels = np.asarray([rows[i]["true_leaf"] for i in keep], dtype=np.int64)
    bank = core.make_bank(f[keep], labels, meta)
    bank["image_hashes"] = np.asarray([rows[i]["image_sha256"] for i in keep], dtype="U64")
    bank["paths"] = np.asarray([rows[i]["path"] for i in keep], dtype=str)
    memory_path = folder / "memory.npz"
    np.savez_compressed(str(memory_path), **bank)
    info = {
        "schema_version": 4, "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "taxonomy": meta,
        "extraction_signature": extraction_signature(cfg),
        "memory_sha256": sha256(memory_path), "logit_scale": scale,
        "source_splits": ["train"], "train_list_sha256": sha256(resolve(cfg["data"]["train"])),
        "all_train_image_hashes": sorted(by_hash),
        "training_image_count": len(rows), "duplicate_training_images_removed": len(rows) - len(unique),
        "memory_image_count": len(keep), "feature_dimension": int(f.shape[1]),
        "memory_counts": {meta["leaf_names"][c]: int(np.sum(labels == c)) for c in range(len(meta["leaf_names"]))},
        "model_updated": False, "transform": "deterministic evaluation transform",
    }
    write_json(folder / "memory.json", info)
    if _residual_enabled(settings):
        print("Fitting train-only cross-fit leave-one-species-out residual support...")
        state = residual.fit(bank, meta, settings["residual"])
        write_json(folder / "residual_state.json", {
            "schema_version": 1, "memory_sha256": info["memory_sha256"],
            "settings_fingerprint": _settings_fingerprint(settings),
            "implementation_signature": _implementation_signature(), "state": state})
        print("Train episode selection: {}".format(state["selection"]["chosen"]))
    print("Memory ready: {} images from {} train records".format(len(keep), len(rows)))
    print("memory: {}".format(memory_path))
    print("No val/test images were loaded. No model parameters were updated.")


def _collect(cfg, splits, model, texts, meta, device, forbidden):
    all_rows, features, pc, lc = [], [], [], []
    seen = set(forbidden)
    for split in splits:
        rows, f, p, l = extract(cfg, split, model, texts, meta, device)
        assert_disjoint(rows, seen, split)
        seen.update(r["image_sha256"] for r in rows)
        all_rows.extend(rows); features.append(f); pc.append(p); lc.append(l)
    return all_rows, np.concatenate(features), np.concatenate(pc), np.concatenate(lc)


def _retrieve(features, bank, meta, settings):
    return core.retrieve(features, bank, meta,
                         k=int(settings.get("child_k", 3)),
                         root_k=int(settings.get("root_k", 10)),
                         chunk_size=int(settings.get("query_chunk_size", 128)))


def _metrics(rows):
    # Existing v3 metric definitions are retained so comparisons are meaningful.
    from .metrics import evaluate_open_set
    groups = {s: [r for r in rows if r["status"] == s] for s in ("known", "intra", "extra")}
    metrics = evaluate_open_set(groups["known"], groups["intra"], groups["extra"])
    known = groups["known"]
    metrics["known"]["raw_text_parent_accuracy"] = float(np.mean([
        r["text_pred_parent"] == r["true_parent"] for r in known]))
    metrics["known"]["closed_hca"] = float(np.mean([
        r["text_pred_parent"] == r["true_parent"] and r["global_pred_leaf"] == r["true_leaf"]
        for r in known]))
    metrics["known"]["routed_joint_accuracy_before_rejection"] = float(np.mean([
        r["candidate_parent"] == r["true_parent"] and r["candidate_leaf"] == r["true_leaf"]
        for r in known]))
    metrics["known"]["closed_hca_definition"] = "raw text parent AND raw global text leaf, before cache routing/rejection"
    metrics["fine_grained_detection"] = residual.diagnostics(rows)
    return metrics


def _summary(metrics):
    return {
        "known_parent_accuracy": metrics["known"]["parent_accuracy"],
        "known_end_to_end_leaf_accuracy": metrics["known"]["end_to_end_leaf_accuracy"],
        "known_leaf_coverage": metrics["known"]["known_leaf_coverage"],
        "intra_cfr": metrics["intra"]["correct_fallback_rate"],
        "intra_oser": metrics["intra"]["over_specification_error_rate"],
        "intra_global_rejection_rate": metrics["intra"]["intra_global_rejection_rate"],
        "extra_far": metrics["extra"]["false_parent_acceptance_rate"],
        "extra_auroc": metrics["extra"]["auroc"],
        "drta": metrics["overall"]["deepest_reliable_taxon_accuracy"],
        "open_world_leaf_precision": metrics["overall"].get("open_world_accepted_leaf_precision"),
        "intra_macro_parent_auroc": metrics["fine_grained_detection"]["macro_parent_auroc"],
        "intra_macro_parent_species_auroc": metrics["fine_grained_detection"]["macro_parent_species_auroc"],
    }


def calibrate():
    args = arguments("calibration")
    cfg, run, checkpoint, folder = _setup(args)
    outputs = [folder / "calibration.json", folder / "validation_report.json",
               folder / "validation_scores.jsonl"]
    _protect(outputs, args.overwrite)
    bank, bank_info = load_bank(folder, cfg, checkpoint)
    settings = cfg["visual_support"]
    state = _load_residual(folder, bank_info, settings) if _residual_enabled(settings) else None
    model, texts, meta, device, scale = load_model(cfg, checkpoint)
    if meta != bank_info["taxonomy"]:
        raise ValueError("Memory taxonomy changed")
    splits = ("val_known", "val_intra", "val_extra")
    rows, f, p, l = _collect(cfg, splits, model, texts, meta, device,
                           bank_info["all_train_image_hashes"])
    fit, cal = core.validation_partition(rows, float(settings.get("fit_fraction", 0.5)),
                                        int(settings.get("seed", 41)))
    support = _retrieve(f, bank, meta, settings)
    routing, routing_report = core.routing_fit(p, l, support, meta, scale, rows, fit, settings)
    e = core.evidence(p, l, support, meta, scale, routing)
    calibration = core.fit_calibration(e, rows, meta, fit, cal, settings)
    rs = None
    if state is not None:
        rs = residual.score(f, bank, state, e["pred_parent"], int(settings.get("query_chunk_size", 128)))
        calibration["residual_calibration"] = residual.calibrate(rs, rows, meta, cal, settings)
        calibration["schema_version"] = _calibration_schema(settings)
        v4_out = _baseline_evidence(core.predict(e, rows, calibration, "coverage"))
        calibration["matched_v4_calibration"] = residual.calibrate(v4_out, rows, meta, cal, settings)
        calibration["residual_calibration"]["empirical_profile_constraints"] = {}
        calibration["residual_state_sha256"] = sha256(folder / "residual_state.json")
    calibration.update({
        "routing": routing, "routing_selection": routing_report,
        "metadata": {
            "checkpoint_sha256": bank_info["checkpoint_sha256"],
            "memory_sha256": bank_info["memory_sha256"],
            "extraction_signature": bank_info["extraction_signature"],
            "settings": settings, "settings_fingerprint": _settings_fingerprint(settings),
            "calibration_splits": list(splits),
            "validation_image_hashes": [r["image_sha256"] for r in rows],
            "fit_image_hashes": [rows[i]["image_sha256"] for i in fit],
            "threshold_image_hashes": [rows[i]["image_sha256"] for i in cal],
            "validation_list_sha256": {s: sha256(resolve(cfg["data"][s])) for s in splits},
            "logit_scale": scale,
            "primary_profile": settings.get("primary_profile", "coverage"),
            "group_warning": "Splits are image-disjoint within observed validation species, not independent novel species",
            "no_unseen_risk_guarantee": True,
        },
    })
    report = {"routing": routing, "root_method": calibration["root_method"],
              "child_method": calibration["child_method"], "fit_count": len(fit),
              "threshold_count": len(cal), "profiles": {}}
    if state is not None:
        report["baseline_v4_child_method"] = report["child_method"]
        report["child_method"] = {"name": calibration["residual_calibration"]["method_name"]}
        report["residual_training_selection"] = state["selection"]
        report["residual_branch_calibration"] = calibration["residual_calibration"]
        report["baseline_v4_profiles"] = {}
        report["matched_calibration_v4_profiles"] = {}
    serial = []
    requested_profiles = "all" if _balanced_enabled(settings) else "both"
    profile_pairs = (_profile_pairs(settings, requested_profiles) if state is not None
                     else [("coverage", "coverage"), ("risk", "risk")])
    for profile, root_profile in profile_pairs:
        baseline = core.predict(e, rows, calibration, root_profile)
        predicted = (_refine(baseline, rs, calibration["residual_calibration"], meta, profile,
                             calibration["residual_calibration"]["method_name"])
                     if state is not None else baseline)
        if state is not None:
            constraints = _source_constraints(predicted, cal, settings)
            calibration["residual_calibration"]["empirical_profile_constraints"][profile] = constraints
            report["baseline_v4_profiles"][profile] = {}
            matched = _refine(baseline, v4_out, calibration["matched_v4_calibration"], meta, profile,
                              "v4_score_parent_calibration")
            report["matched_calibration_v4_profiles"][profile] = {}
        for record in predicted:
            record["support_neighbor_path"] = str(bank["paths"][record["support_neighbor_index"]])
        report["profiles"][profile] = {}
        for subset, indices in (("selection_fit", fit), ("threshold_calibration", cal),
                                ("all_validation", np.arange(len(rows)))):
            m = _metrics([predicted[i] for i in indices])
            report["profiles"][profile][subset] = m
            if state is not None:
                report["baseline_v4_profiles"][profile][subset] = _metrics([baseline[i] for i in indices])
                report["matched_calibration_v4_profiles"][profile][subset] = _metrics([matched[i] for i in indices])
        fit_set = set(fit.tolist())
        for i, r in enumerate(predicted):
            r["validation_partition"] = "selection_fit" if i in fit_set else "threshold_calibration"
            serial.append(r)
    write_json(folder / "calibration.json", calibration)
    write_json(folder / "validation_report.json", report)
    write_records(folder / "validation_scores.jsonl", serial)
    print("Visual-support calibration complete; NO test split was loaded.")
    print("routing: {}".format(routing))
    print("root score: {}; child score: {}".format(calibration["root_method"]["name"], report["child_method"]["name"]))
    for profile, _ in profile_pairs:
        print(profile + " / threshold calibration:")
        print(json.dumps(_summary(report["profiles"][profile]["threshold_calibration"]), indent=2))
        constraints = (calibration["residual_calibration"]["empirical_profile_constraints"][profile]
                       if state is not None else calibration["profiles"][profile])
        print("Empirical source constraints: {}".format(constraints["empirical_constraints_satisfied"]))
    print("WARNING: validation risk limits do NOT guarantee risk on unseen species.")
    print("Report: {}".format(folder / "validation_report.json"))


def test():
    args = arguments("test")
    cfg, run, checkpoint, folder = _setup(args)
    calibration_path = folder / "calibration.json"
    calibration = read_json(calibration_path)
    expected_schema = _calibration_schema(cfg["visual_support"])
    if calibration.get("schema_version") != expected_schema:
        raise ValueError("Calibration schema/config mismatch; expected {}".format(expected_schema))
    cm = calibration["metadata"]
    if tuple(cm["calibration_splits"]) != ("val_known", "val_intra", "val_extra"):
        raise ValueError("Calibration provenance is invalid")
    if cm["settings_fingerprint"] != _settings_fingerprint(cfg["visual_support"]):
        raise ValueError("Visual configuration changed after calibration; do not tune settings on test")
    bank, bank_info = load_bank(folder, cfg, checkpoint)
    state = (_load_residual(folder, bank_info, cfg["visual_support"])
             if _residual_enabled(cfg["visual_support"]) else None)
    if state is not None and sha256(folder / "residual_state.json") != calibration["residual_state_sha256"]:
        raise ValueError("Residual state changed after calibration")
    for key in ("checkpoint_sha256", "memory_sha256", "extraction_signature"):
        if cm[key] != bank_info[key]:
            raise ValueError("Calibration/memory {} mismatch".format(key))
    profile_pairs = _profile_pairs(cfg["visual_support"], args.profiles)
    profiles = [profile for profile, _ in profile_pairs]
    output = folder / "test"
    paths = [output / "comparison.json"]
    for profile in profiles:
        paths.extend([output / profile / "metrics.json", output / profile / "predictions.jsonl"])
        if state is not None:
            paths.extend([output / profile / "baseline_v4_metrics.json", output / profile / "baseline_v4_predictions.jsonl"])
            paths.extend([output / profile / "matched_v4_metrics.json", output / profile / "matched_v4_predictions.jsonl"])
    _protect(paths, args.overwrite)
    model, texts, meta, device, scale = load_model(cfg, checkpoint)
    if meta != calibration["taxonomy"] or abs(scale - cm["logit_scale"]) > 1e-5:
        raise ValueError("Taxonomy/logit scale changed after calibration")
    # Test only reads test images + frozen train-memory/validation artifacts.
    # It does NOT instantiate train or validation datasets.
    forbidden = set(bank_info["all_train_image_hashes"]) | set(cm["validation_image_hashes"])
    rows, f, p, l = _collect(cfg, ("test_known", "test_intra", "test_extra"),
                           model, texts, meta, device, forbidden)
    support = _retrieve(f, bank, meta, cm["settings"])
    e = core.evidence(p, l, support, meta, scale, calibration["routing"])
    rs = (residual.score(f, bank, state, e["pred_parent"], int(cm["settings"].get("query_chunk_size", 128)))
          if state is not None else None)
    comparison = {"primary_profile": cm["primary_profile"], "profiles": {},
                  "note": "All requested frozen profiles share ONE feature pass. No test fitting."}
    for profile, root_profile in profile_pairs:
        baseline = core.predict(e, rows, calibration, root_profile)
        predicted = (_refine(baseline, rs, calibration["residual_calibration"], meta, profile,
                             calibration["residual_calibration"]["method_name"])
                     if state is not None else baseline)
        for record in predicted:
            record["support_neighbor_path"] = str(bank["paths"][record["support_neighbor_index"]])
        m = _metrics(predicted)
        constraint_report = (calibration["residual_calibration"]["empirical_profile_constraints"][profile]
                             if state is not None else calibration["profiles"][profile])
        m["metadata"] = {"checkpoint_sha256": cm["checkpoint_sha256"],
                         "memory_sha256": cm["memory_sha256"],
                         "calibration_sha256": sha256(calibration_path),
                         "profile": profile, "root_method": calibration["root_method"],
                         "child_method": calibration["child_method"],
                         "routing": calibration["routing"],
                         "test_splits": ["test_known", "test_intra", "test_extra"],
                         "root_profile": root_profile,
                         "calibration_source_constraints_satisfied": constraint_report["empirical_constraints_satisfied"],
                         "no_population_safety_guarantee": True}
        if state is not None:
            m["metadata"]["child_method"] = calibration["residual_calibration"]["method_name"]
            m["metadata"]["residual_state_sha256"] = calibration["residual_state_sha256"]
            m["metadata"]["root_identical_to_paired_v4_baseline"] = True
            m["metadata"]["calibration_source_constraints_satisfied"] = calibration["residual_calibration"]["empirical_profile_constraints"][profile]["empirical_constraints_satisfied"]
        profile_dir = output / profile
        write_json(profile_dir / "metrics.json", m)
        write_records(profile_dir / "predictions.jsonl", predicted)
        comparison["profiles"][profile] = _summary(m)
        if state is not None:
            for record in baseline:
                record["support_neighbor_path"] = str(bank["paths"][record["support_neighbor_index"]])
            baseline_metrics = _metrics(baseline)
            write_json(profile_dir / "baseline_v4_metrics.json", baseline_metrics)
            write_records(profile_dir / "baseline_v4_predictions.jsonl", baseline)
            comparison.setdefault("paired_baseline_v4", {})[profile] = _summary(baseline_metrics)
            matched = _refine(baseline, _baseline_evidence(baseline), calibration["matched_v4_calibration"],
                              meta, profile, "v4_score_parent_calibration")
            matched_metrics = _metrics(matched)
            write_json(profile_dir / "matched_v4_metrics.json", matched_metrics)
            write_records(profile_dir / "matched_v4_predictions.jsonl", matched)
            comparison.setdefault("matched_calibration_v4", {})[profile] = _summary(matched_metrics)
            comparison["root_decisions_identical"] = True
    write_json(output / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))
    print("Test results: {}".format(output))
