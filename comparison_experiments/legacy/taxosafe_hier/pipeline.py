"""Frozen checkpoint -> spatial cache -> adapter training -> calibration -> test.

The root memory/calibration are verified outputs of the original TaxoSafe run.
This extension never fits on test features and never changes the original run.
"""
import copy
from pathlib import Path
import numpy as np
import yaml
from taxosafe_visual import core, residual
from taxosafe_visual.runtime import (resolve, sha256, read_json, write_json,
                                    write_records, load_model, assert_disjoint)
from taxosafe_visual.pipeline import _metrics, _summary
from .calibration import protected_thresholds


def read_rows(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cache(folder, name):
    base = Path(folder) / "cache"
    with np.load(str(base / (name + ".npz")), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    rows = read_rows(base / (name + ".jsonl"))
    if len(rows) != len(arrays["global"]) or len(rows) != len(arrays["patches"]):
        raise ValueError("Cache/manifest lengths differ")
    return arrays, rows


def original_bank(plan):
    with np.load(str(resolve(plan["root_memory"])), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def memory_subset(plan, data, rows):
    # Same images and order as original TaxoSafe memory: no support-size advantage.
    hashes = original_bank(plan)["image_hashes"].astype(str).tolist()
    lookup = {row["image_sha256"]: i for i, row in enumerate(rows)}
    if len(lookup) != len(rows) or any(h not in lookup for h in hashes):
        raise ValueError("Reference memory is not a unique subset of training cache")
    ids = np.asarray([lookup[h] for h in hashes], dtype=np.int64)
    subset = {k: data[k][ids] for k in ("global", "patches")}
    labels = np.asarray([rows[i]["true_leaf"] for i in ids], dtype=np.int64)
    if not np.array_equal(labels, original_bank(plan)["labels"]):
        raise ValueError("Reference memory label mapping changed")
    return subset, labels


def cache(plan, name):
    from .extraction import extract_spatial
    cfg = yaml.safe_load(resolve(plan["training_config"]).read_text(encoding="utf-8-sig"))
    folder = resolve(plan["suite"])
    model, texts, meta, device, scale = load_model(cfg, resolve(plan["checkpoint"]))
    if meta != plan["taxonomy"]:
        raise ValueError("Taxonomy differs from the original training run")
    splits = {"train": ("train",), "val": ("val_known", "val_intra", "val_extra"),
              "test": ("test_known", "test_intra", "test_extra")}[name]
    forbidden = set()
    for previous in ({"train": (), "val": ("train",), "test": ("train", "val")}[name]):
        _, rows = load_cache(folder, previous)
        forbidden.update(r["image_sha256"] for r in rows)
    rows, arrays = extract_spatial(cfg, splits, model, texts, meta, device, forbidden,
                                   plan["settings"]["spatial_pool_side"])
    (folder / "cache").mkdir(exist_ok=True)
    np.savez_compressed(str(folder / "cache" / (name + ".npz")), **arrays)
    write_records(folder / "cache" / (name + ".jsonl"), rows)


def train(plan, device):
    import torch
    from .model import train_adapter, save_adapter
    torch.set_num_threads(4)
    folder = resolve(plan["suite"])
    data, rows = load_cache(folder, "train")
    for variant in plan["settings"]["variants"]:
        path = folder / "artifacts" / variant
        path.mkdir(parents=True, exist_ok=True)
        model, history = train_adapter(data, rows, plan["taxonomy"], plan["settings"]["adapter"], variant, plan["seed"], device)
        save_adapter(path / "adapter.npz", model)
        write_json(path / "training.json", {"variant": variant, "seed": plan["seed"],
                   "source_splits": ["train"], "epochs": history,
                   "train_cache_sha256": sha256(folder / "cache/train.npz"),
                   "backbone_updated": False, "adapter_updated": variant != "identity",
                   "trainable_parameters": sum(p.numel() for p in model.parameters()) if variant != "identity" else 0,
                   "selection": "fixed final epoch; no validation/test checkpoint selection",
                   "pseudo_species_seen_by_encoder": True,
                   "singleton_note": "No within-parent pseudo-unknown episode is possible for singleton branches"})


def frozen_root(plan, query, rows):
    calibration = read_json(resolve(plan["root_calibration"]))
    settings = calibration["metadata"]["settings"]
    support = core.retrieve(query["global"], original_bank(plan), plan["taxonomy"],
                            k=int(settings.get("child_k", 3)), root_k=int(settings.get("root_k", 10)),
                            chunk_size=int(settings.get("query_chunk_size", 128)))
    e = core.evidence(query["parent_cosine"], query["leaf_cosine"], support, plan["taxonomy"],
                      calibration["metadata"]["logit_scale"], calibration["routing"])
    return core.predict(e, rows, calibration, "coverage")


def scores(plan, query, train_data, train_rows, root, variant, device):
    from .model import load_adapter, score_adapter
    folder = resolve(plan["suite"])
    model = load_adapter(folder / "artifacts" / variant / "adapter.npz", train_data,
                         plan["taxonomy"], plan["settings"]["adapter"], variant, device)
    support, labels = memory_subset(plan, train_data, train_rows)
    return score_adapter(model, query, support, labels,
                         [r["candidate_parent"] for r in root], plan["settings"]["query_chunk_size"])


def settings_for_thresholds(plan):
    settings = copy.deepcopy(read_json(resolve(plan["root_calibration"]))["metadata"]["settings"])
    settings["residual"] = {"threshold_mode": "hierarchical_shrinkage", "threshold_shrinkage": "auto",
                            "balanced_profile": True}
    return settings


def apply(plan, root, out, calibration, profile, variant):
    predictions = residual.apply(root, out, calibration, plan["taxonomy"], profile, "hier_local_evidence_" + variant)
    for before, after in zip(root, predictions):
        for field in ("candidate_parent", "root_knownness_score", "root_gate_margin"):
            if before[field] != after[field]:
                raise AssertionError("Frozen root changed")
        if (before["prediction_type"] == "global_unknown") != (after["prediction_type"] == "global_unknown"):
            raise AssertionError("Frozen root rejection changed")
    return predictions


def validation_indices(plan, rows):
    old = read_json(resolve(plan["root_calibration"]))["metadata"]
    hashes = {row["image_sha256"]: i for i, row in enumerate(rows)}
    if set(hashes) != set(old["validation_image_hashes"]):
        raise ValueError("Validation contents differ from frozen root calibration")
    fit = set(old["fit_image_hashes"])
    held = set(old["threshold_image_hashes"])
    if fit & held or fit | held != set(hashes):
        raise ValueError("Invalid original validation partition")
    return [hashes[h] for h in old["threshold_image_hashes"]]


def audit_metrics(rows):
    m = _metrics(rows)
    known = [r for r in rows if r["status"] == "known"]
    leaves = sorted({r["true_leaf"] for r in known})
    correct = lambda r: r["candidate_leaf"] == r["true_leaf"] and r["candidate_parent"] == r["true_parent"]
    m["known"]["routed_macro_leaf_accuracy_before_rejection"] = float(np.mean([
        np.mean([correct(r) for r in known if r["true_leaf"] == c]) for c in leaves])) if leaves else None
    from sklearn.metrics import f1_score
    m["known"]["routed_macro_f1_before_rejection"] = float(f1_score(
        [r["true_leaf"] for r in known], [r["candidate_leaf"] for r in known], average="macro", zero_division=0)) if known else None
    # OSCR over raw child scores and routed correct species, before root/child
    # thresholds. This is a near-open curve, not the full hierarchical policy.
    near = [r for r in rows if r["status"] in ("known", "intra")]
    nk = sum(r["status"] == "known" for r in near)
    nu = len(near) - nk
    if nk and nu:
        values = np.asarray([r["child_knownness_score"] for r in near])
        order = np.argsort(-values, kind="stable")
        good = np.asarray([r["status"] == "known" and correct(r) for r in near], dtype=int)[order]
        unknown = np.asarray([r["status"] == "intra" for r in near], dtype=int)[order]
        ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(order) - 1]
        ccr = np.r_[0., np.cumsum(good)[ends] / nk]
        fpr = np.r_[0., np.cumsum(unknown)[ends] / nu]
        m["near_oscr_before_root_gate"] = float(np.sum(np.diff(fpr) * (ccr[1:] + ccr[:-1]) / 2))
    else:
        m["near_oscr_before_root_gate"] = None
    m["near_oscr_definition"] = "Area under routed correct-known rate vs intra false acceptance, sweeping raw child score; root gate excluded"
    # This is only a descriptive count error for this supplied image mixture.
    # Real abundance/diversity needs acquisition groups and sampling volumes.
    classes = max([r["candidate_leaf"] for r in rows] + leaves) + 1
    truth, predicted = np.zeros(classes + 1), np.zeros(classes + 1)
    for r in rows:
        truth[r["true_leaf"] if r["status"] == "known" else classes] += 1
        predicted[r["candidate_leaf"] if r["prediction_type"] == "known" else classes] += 1
    m["descriptive_count_error"] = {
        "known_leaf_plus_unresolved_total_variation": float(np.abs(truth - predicted).sum() / (2 * len(rows))) if rows else None,
        "truth": truth.tolist(), "prediction": predicted.tolist(),
        "scope": "This artificial image mixture only, not measured ecological abundance or an independent community benchmark"}
    # Existing global_pred_leaf remains RAW text prediction. Do not mislabel it
    # as the trained hierarchical classifier's closed-set result.
    return m


def compact_metrics(m):
    r = _summary(m)
    r["closed_routed_accuracy"] = m["known"]["routed_joint_accuracy_before_rejection"]
    r["closed_routed_macro_accuracy"] = m["known"]["routed_macro_leaf_accuracy_before_rejection"]
    r["closed_routed_macro_f1"] = m["known"]["routed_macro_f1_before_rejection"]
    r["near_oscr_before_root_gate"] = m["near_oscr_before_root_gate"]
    return r


def calibrate(plan, device):
    folder = resolve(plan["suite"])
    train_data, train_rows = load_cache(folder, "train")
    query, rows = load_cache(folder, "val")
    assert_disjoint(rows, [r["image_sha256"] for r in train_rows], "validation")
    ids = validation_indices(plan, rows)
    root = frozen_root(plan, query, rows)
    settings = settings_for_thresholds(plan)
    identity_scores = scores(plan, query, train_data, train_rows, root, "identity", device)
    identity_cal = residual.calibrate(identity_scores, rows, plan["taxonomy"], ids, settings)
    identity_reference = apply(plan, root, identity_scores, identity_cal, "balanced", "identity")
    report = {"source": "held validation only", "test_loaded": False,
              "reference": "identity/balanced: same support, original root, unadapted 1NN",
              "threshold_images": len(ids), "variants": {}}
    for variant in plan["settings"]["variants"]:
        out = identity_scores if variant == "identity" else scores(plan, query, train_data, train_rows, root, variant, device)
        cal = residual.calibrate(out, rows, plan["taxonomy"], ids, settings)
        protected = protected_thresholds(out, rows, identity_reference, root, plan["taxonomy"], ids)
        cal["profiles"]["protected"] = protected["thresholds"]
        cal["protected_report"] = protected
        path = folder / "artifacts" / variant
        write_json(path / "calibration.json", cal)
        report["variants"][variant] = {"protected_floors_feasible": protected["all_known_floors_feasible"], "profiles": {}}
        for profile in ("coverage", "balanced", "protected"):
            predictions = apply(plan, root, out, cal, profile, variant)
            selected = [predictions[i] for i in ids]
            report["variants"][variant]["profiles"][profile] = compact_metrics(audit_metrics(selected))
    write_json(folder / "validation_report.json", report)
    print("Validation report:", folder / "validation_report.json")
    print("Known utility floor feasible:", report["variants"]["full"]["protected_floors_feasible"])


def test(plan, device):
    folder = resolve(plan["suite"])
    train_data, train_rows = load_cache(folder, "train")
    _, validation_rows = load_cache(folder, "val")
    query, rows = load_cache(folder, "test")
    if any(not r["split"].startswith("test_") for r in rows):
        raise ValueError("Final evaluation requires test records")
    assert_disjoint(rows, [r["image_sha256"] for r in train_rows + validation_rows], "test")
    root = frozen_root(plan, query, rows)
    summary = {"seed": plan["seed"], "primary_variant": "full", "primary_profile": "protected",
               "protocol": "DEVELOPMENT: existing test previously inspected; not confirmatory SOTA",
               "root_comparison": "All variants use bit-identical original coverage root",
               "rows": []}
    for variant in plan["settings"]["variants"]:
        out = scores(plan, query, train_data, train_rows, root, variant, device)
        base = folder / "artifacts" / variant
        cal = read_json(base / "calibration.json")
        for profile in ("coverage", "balanced", "protected"):
            predictions = apply(plan, root, out, cal, profile, variant)
            path = base / "test" / profile
            path.mkdir(parents=True)
            m = audit_metrics(predictions)
            write_json(path / "metrics.json", m)
            write_records(path / "predictions.jsonl", predictions)
            summary["rows"].append({"method": variant, "profile": profile, **compact_metrics(m)})
    write_json(folder / "summary.json", summary)
