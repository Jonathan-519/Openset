"""Development-only, two-offset whole-tree calibration.

The fixed output order is root, all parents, then all leaves. Root has bias zero;
only parent_bias and leaf_bias are fitted. A finite grid failure is evidence about
that grid, not a proof that no continuous operating point exists. No image, model,
support statistic, feature weight or class-specific threshold is fitted here.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json

import numpy as np

SCHEMA_VERSION = "support_v1"
TARGETS = {
    "known_end_to_end_leaf_accuracy": {"threshold": .90, "operator": ">"},
    "intra_correct_fallback_rate": {"threshold": .85, "operator": ">="},
    "extra_global_unknown_recall": {"threshold": .90, "operator": ">"},
    "open_world_accepted_leaf_precision": {"threshold": .90, "operator": ">"},
}
STATUSES = ("known", "intra", "extra")
KINDS = ("global_unknown", "intra_unknown", "known")


def _hierarchy(meta):
    p, c = len(meta["parent_names"]), len(meta["leaf_names"])
    mapping = np.asarray(meta["leaf_to_parent"], dtype=int)
    if p < 1 or c < 1 or mapping.shape != (c,) or set(mapping.tolist()) != set(range(p)):
        raise ValueError("Each parent must have at least one leaf; invalid hierarchy")
    return p, c, mapping


def _digest(row):
    keys = ("image_sha256", "content_sha256", "sha256")
    values = [str(row[k]).lower() for k in keys if row.get(k) is not None]
    if not values or len(set(values)) != 1:
        raise ValueError("Missing or conflicting content SHA-256 identity")
    value = values[0]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid content SHA-256 identity")
    return value


def unique_records(records):
    """Deduplicate content, rejecting conflicting annotations or predictions.

    Path aliases are allowed within one split. A shared hash across splits or
    sources is an audit error, even if the numerical labels happen to match.
    """
    unique, seen = [], {}
    fields = ("split", "status", "true_parent", "true_leaf", "source", "species")
    prediction_fields = ("prediction_type", "parent", "leaf", "candidate_parent", "candidate_leaf")
    for row in records:
        key = _digest(row)
        if key in seen:
            old = seen[key]
            if any(old.get(k) != row.get(k) for k in fields + prediction_fields):
                raise ValueError("Conflicting annotations/predictions or cross-split content: " + key)
            if "log_probs" in old or "log_probs" in row:
                if "log_probs" not in old or "log_probs" not in row or not np.array_equal(old["log_probs"], row["log_probs"]):
                    raise ValueError("Same content has inconsistent model evidence: " + key)
            continue
        seen[key] = row
        unique.append(row)
    return unique


def _scores(records, meta):
    p, c, mapping = _hierarchy(meta)
    if not records:
        return np.empty((0, 1 + p + c)), p, c, mapping
    values = np.asarray([r["log_probs"] for r in records], dtype=np.float64)
    if values.shape != (len(records), 1 + p + c) or not np.isfinite(values).all():
        raise ValueError("Full-support log_probs must be finite and follow root,parent,leaf order")
    top = values.max(axis=1)
    log_total = top + np.log(np.exp(values - top[:, None]).sum(axis=1))
    if not np.allclose(log_total, 0., atol=1e-4, rtol=0.):
        raise ValueError("log_probs must describe a normalized whole-tree distribution")
    return values, p, c, mapping


def raw_records(rows, evidence_output, leaf_logits, meta):
    """Attach model probabilities; labels are never used to produce evidence."""
    values = evidence_output["log_probs"] if isinstance(evidence_output, dict) else evidence_output
    values = np.asarray(values, dtype=np.float64)
    p, c, _ = _hierarchy(meta)
    if values.shape != (len(rows), 1 + p + c):
        raise ValueError("Evidence and metadata shapes differ")
    original = None if leaf_logits is None else np.asarray(leaf_logits)
    if original is not None and (original.shape != (len(rows), c) or not np.isfinite(original).all()):
        raise ValueError("Invalid closed-set leaf logits")
    result = []
    for i, row in enumerate(rows):
        record = dict(row)
        record["log_probs"] = values[i].tolist()
        if original is not None:
            record["global_pred_leaf"] = int(original[i].argmax())
        result.append(record)
    _scores(result, meta)
    return result


def _biases(state, meta):
    if state.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Support calibration schema mismatch")
    if "meta" in state and state["meta"] != meta:
        raise ValueError("Calibration hierarchy differs from inference hierarchy")
    values = float(state["parent_bias"]), float(state["leaf_bias"])
    if not np.isfinite(values).all():
        raise ValueError("Calibration biases must be finite")
    return values


def decode_records(records, state, meta):
    """Decode the entire tree jointly, preserving every original record.

    Ties deterministically prefer root, then parent, then leaf, and within a
    depth the lowest node index. Candidate fields remain populated at root for
    compatibility with metrics_open; parent/leaf identify the actual output.
    """
    records = list(records)
    values, p, c, mapping = _scores(records, meta)
    pb, lb = _biases(state, meta)
    shifted = values.copy()
    shifted[:, 1:1 + p] += pb
    shifted[:, 1 + p:] += lb
    outputs = shifted.argmax(axis=1) if records else []
    result = []
    for i, (row, node) in enumerate(zip(records, outputs)):
        best_leaf = int(values[i, 1 + p:].argmax())
        best_parent = int(values[i, 1:1 + p].argmax())
        if node == 0:
            kind, parent, leaf = "global_unknown", None, None
            candidate_parent = best_parent
            children = np.flatnonzero(mapping == candidate_parent)
            candidate_leaf = int(children[values[i, 1 + p + children].argmax()])
        elif node <= p:
            kind, parent, leaf = "intra_unknown", int(node - 1), None
            candidate_parent = parent
            children = np.flatnonzero(mapping == parent)
            candidate_leaf = int(children[values[i, 1 + p + children].argmax()])
        else:
            leaf = int(node - 1 - p)
            parent = candidate_parent = int(mapping[leaf])
            kind, candidate_leaf = "known", leaf
        probability = np.exp(values[i])
        parent_mass = probability[1 + candidate_parent]
        leaf_mass = probability[1 + p:][mapping == candidate_parent].sum()
        local = float(leaf_mass / max(parent_mass + leaf_mass, np.finfo(float).tiny))
        record = dict(row)
        record.update({
            "prediction_type": kind, "output_node": int(node), "parent": parent, "leaf": leaf,
            "candidate_parent": candidate_parent, "candidate_leaf": candidate_leaf,
            "candidate_parent_name": meta["parent_names"][candidate_parent],
            "candidate_leaf_name": meta["leaf_names"][candidate_leaf],
            "true_parent_name": None if row.get("true_parent") is None else meta["parent_names"][int(row["true_parent"])],
            "true_leaf_name": None if row.get("true_leaf") is None else meta["leaf_names"][int(row["true_leaf"])],
            "global_pred_leaf": int(row.get("global_pred_leaf", best_leaf)),
            "root_knownness_score": float(probability[1:].sum()),
            "local_knownness_score": local, "local_known_margin": local,
            "parent_bias": pb, "leaf_bias": lb,
        })
        result.append(record)
    return result


apply_router = decode_records


def _rate(successes, total):
    return None if total == 0 else float(successes / total)


def _gates(counts):
    names = tuple(TARGETS)
    pairs = ((counts["known_correct"], counts["known"]),
             (counts["intra_correct"], counts["intra"]),
             (counts["extra_correct"], counts["extra"]),
             (counts["known_correct"], counts["leaf_outputs"]))
    # Integer arithmetic makes >90% distinct from >=90%, even at exact boundaries.
    checks, metrics, requirements = {}, {}, {}
    for name, (successes, total) in zip(names, pairs):
        inclusive = TARGETS[name]["operator"] == ">="
        required = (17 * total + 19) // 20 if inclusive else (9 * total) // 10 + 1
        passed = total > 0 and successes >= required
        checks[name] = bool(passed)
        metrics[name] = _rate(successes, total)
        requirements[name] = {"correct": successes, "total": total, "required_correct": required,
                              "missing_correct": max(0, required - successes), **TARGETS[name]}
    return {"metrics": metrics, "checks": checks, "requirements": requirements,
            "targets_passed": bool(all(checks.values())), "counts": counts}


def _count_predictions(records):
    counts = {"known": 0, "intra": 0, "extra": 0, "known_correct": 0,
              "intra_correct": 0, "extra_correct": 0, "leaf_outputs": 0}
    for row in records:
        status, kind = row["status"], row["prediction_type"]
        if status not in STATUSES or kind not in KINDS:
            raise ValueError("Unknown sample status or prediction_type")
        counts[status] += 1
        counts["leaf_outputs"] += int(kind == "known")
        correct = ((kind == "known" and row["leaf"] == row["true_leaf"] and row["parent"] == row["true_parent"])
                   if status == "known" else
                   (kind == "intra_unknown" and row["parent"] == row["true_parent"])
                   if status == "intra" else kind == "global_unknown")
        counts[status + "_correct"] += int(correct)
    return counts


def _group_report(rows, status):
    counts = _count_predictions(rows)
    n = counts[status]
    types = dict(Counter(r["prediction_type"] for r in rows))
    return {"sample_count": n, "correct_count": counts[status + "_correct"],
            "correct_rate": _rate(counts[status + "_correct"], n),
            "leaf_acceptance_rate": _rate(types.get("known", 0), n),
            "parent_fallback_rate": _rate(types.get("intra_unknown", 0), n),
            "root_rejection_rate": _rate(types.get("global_unknown", 0), n),
            "prediction_type_counts": {kind: types.get(kind, 0) for kind in KINDS}}


def evaluate_records(records, meta=None):
    """Four required targets with unique content denominators and group detail."""
    records = list(records)
    rows = unique_records(records)
    report = _gates(_count_predictions(rows))
    report.update({"unit": "unique_image_content_sha256", "input_record_count": len(records),
                   "unique_image_count": len(rows), "duplicate_record_count": len(records) - len(rows),
                   "targets": copy.deepcopy(TARGETS)})
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "known":
            name = row.get("true_leaf_name") or (meta["leaf_names"][int(row["true_leaf"])] if meta else str(row["true_leaf"]))
        else:
            name = row.get("species") or row.get("source") or "unspecified"
        grouped[(row["status"], str(name))].append(row)
    for status, key in (("known", "per_known_leaf"), ("intra", "per_intra_species"), ("extra", "per_extra_source")):
        report[key] = {name: _group_report(group, status)
                       for (group_status, name), group in sorted(grouped.items()) if group_status == status}
    return report


def evaluate_gates(metrics, settings=None):
    """Adapter for metrics_open output; caller must supply deduplicated metrics."""
    if "counts" in metrics and "leaf_outputs" in metrics["counts"]:
        return _gates(metrics["counts"])
    counts = {"known": int(metrics["known"]["sample_count"]),
              "intra": int(metrics["intra"]["sample_count"]),
              "extra": int(metrics["extra"]["sample_count"]),
              "known_correct": int(metrics["overall"]["correct_leaf_count"]),
              "leaf_outputs": int(metrics["overall"]["accepted_leaf_count"])}
    for status, key, denominator in (("intra", "correct_fallback_rate", "intra"),
                                     ("extra", "global_unknown_recall", "extra")):
        value = metrics[status][key]
        counts[status + "_correct"] = 0 if value is None else int(round(float(value) * counts[denominator]))
    return _gates(counts)


def _fit_inputs(known, near, extra, meta):
    rows = []
    for group, status, split in ((known, "known", "val_known"), (near, "intra", "val_intra"), (extra, "extra", "val_extra")):
        group = list(group)
        if not group or any(r.get("split") != split or r.get("status") != status for r in group):
            raise ValueError("Calibration requires nonempty {} records only; test fitting is prohibited".format(split))
        rows.extend(group)
    input_count = len(rows)
    rows = unique_records(rows)
    _, p, c, mapping = _scores(rows, meta)
    for row in rows:
        if row["status"] in ("known", "intra"):
            parent = row.get("true_parent")
            if not isinstance(parent, (int, np.integer)) or not 0 <= int(parent) < p:
                raise ValueError("Known/near records need a valid true_parent")
        if row["status"] == "known":
            leaf = row.get("true_leaf")
            if not isinstance(leaf, (int, np.integer)) or not 0 <= int(leaf) < c or mapping[int(leaf)] != row["true_parent"]:
                raise ValueError("Known leaf/parent annotation mismatch")
    return rows, input_count


def _grid(settings, name):
    if name in settings:
        values = np.asarray(settings[name], dtype=float)
    else:
        count = settings.get("grid_points", 49)
        if isinstance(count, bool) or int(count) != count or not 1 <= count <= 401:
            raise ValueError("grid_points must be an integer in [1,401]")
        values = np.linspace(float(settings.get("bias_min", -6.)), float(settings.get("bias_max", 6.)), int(count))
    if values.ndim != 1 or not 1 <= len(values) <= 401 or not np.isfinite(values).all():
        raise ValueError(name + " must be a finite, nonempty grid of at most 401 offsets")
    if settings.get("bias_min", -6.) > settings.get("bias_max", 6.):
        raise ValueError("bias_min exceeds bias_max")
    return np.unique(values)


def _select(rows, meta, parent_grid, leaf_grid, keep_grid=True):
    values, p, _, mapping = _scores(rows, meta)
    # Maximizing within a depth is unaffected by that depth's scalar offset.
    base = np.c_[values[:, 0], values[:, 1:1 + p].max(1), values[:, 1 + p:].max(1)]
    parent = values[:, 1:1 + p].argmax(1)
    leaf = values[:, 1 + p:].argmax(1)
    status = np.asarray([r["status"] for r in rows])
    known, intra, extra = (status == s for s in STATUSES)
    true_parent = np.asarray([-1 if r.get("true_parent") is None else r["true_parent"] for r in rows])
    true_leaf = np.asarray([-1 if r.get("true_leaf") is None else r["true_leaf"] for r in rows])
    totals = {s: int(np.sum(status == s)) for s in STATUSES}
    best, points, maxima = None, [], {name: 0. for name in TARGETS}
    feasible = 0
    for pb in parent_grid:
        for lb in leaf_grid:
            depth = (base + np.asarray([0., pb, lb])).argmax(1)
            counts = dict(totals)
            counts.update({"known_correct": int(np.sum(known & (depth == 2) & (leaf == true_leaf) & (mapping[leaf] == true_parent))),
                           "intra_correct": int(np.sum(intra & (depth == 1) & (parent == true_parent))),
                           "extra_correct": int(np.sum(extra & (depth == 0))),
                           "leaf_outputs": int(np.sum(depth == 2))})
            report = _gates(counts)
            rates = [report["metrics"][name] or 0. for name in TARGETS]
            deficit = sum(v["missing_correct"] / max(v["total"], 1) for v in report["requirements"].values())
            # Feasibility first; otherwise expose the closest sampled compromise.
            # The objective is fixed, group-balanced, and never learned per source.
            key = (report["targets_passed"], -deficit, float(np.mean(rates)), rates[0],
                   -abs(float(pb)) - abs(float(lb)), -float(pb), -float(lb))
            if best is None or key > best[0]:
                best = key, float(pb), float(lb), report
            feasible += int(report["targets_passed"])
            for name, rate in report["metrics"].items():
                maxima[name] = max(maxima[name], 0. if rate is None else rate)
            if keep_grid:
                points.append({"parent_bias": float(pb), "leaf_bias": float(lb),
                               **report["metrics"], "targets_passed": report["targets_passed"]})
    return {"parent_bias": best[1], "leaf_bias": best[2], "report": best[3],
            "sampled_feasible_count": feasible, "sampled_point_count": len(parent_grid) * len(leaf_grid),
            "individual_sampled_maxima": maxima, "points": points}


def source_loo(known, near, extra, meta, settings=None):
    """Refit only two biases excluding a development source, then score it.

    Near species and extra sources are considered separately. No held-source
    scores enter candidate generation, selection, or any model update. Fixed
    explicit grids also avoid indirectly selecting candidates from the holdout.
    """
    settings = dict(settings or {})
    rows, _ = _fit_inputs(known, near, extra, meta)
    pg, lg = _grid(settings, "parent_bias_grid"), _grid(settings, "leaf_bias_grid")
    folds, skipped = [], []
    for status in ("intra", "extra"):
        sources = sorted({str(r.get("source", "unspecified")) for r in rows if r["status"] == status})
        if len(sources) < 2:
            skipped.append({"status": status, "reason": "At least two development sources are required"})
            continue
        for held in sources:
            holdout = [r for r in rows if r["status"] == status and str(r.get("source", "unspecified")) == held]
            fitted = [r for r in rows if not (r["status"] == status and str(r.get("source", "unspecified")) == held)]
            result = _select(fitted, meta, pg, lg, keep_grid=False)
            state = {"parent_bias": result["parent_bias"], "leaf_bias": result["leaf_bias"]}
            decoded = decode_records(holdout, state, meta)
            folds.append({"status": status, "held_source": held, **state,
                          "fit_count": len(fitted), "fit_targets_passed": result["report"]["targets_passed"],
                          "fit_sources": sorted({str(r.get("source", "unspecified")) for r in fitted if r["status"] == status}),
                          "held_metrics": _group_report(decoded, status)})
    return {"available": bool(folds), "folds": folds, "skipped": skipped,
            "use": "development diagnostic only; never used to change model, weights, or final selected biases"}


def calibrate(known, near, extra, meta, settings=None):
    """Fit two development biases and report whether all fixed targets hold.

    fit_completed means the finite grid search ran. It is deliberately independent
    of targets_passed. A failed target never silently lowers its success threshold.
    """
    settings = dict(settings or {})
    known, near, extra = list(known), list(near), list(extra)
    rows, input_count = _fit_inputs(known, near, extra, meta)
    pg, lg = _grid(settings, "parent_bias_grid"), _grid(settings, "leaf_bias_grid")
    selected = _select(rows, meta, pg, lg)
    state = {"schema_version": SCHEMA_VERSION, "meta": copy.deepcopy(meta),
             "parent_bias": selected["parent_bias"], "leaf_bias": selected["leaf_bias"],
             "root_bias": 0., "fit_completed": True,
             "targets_passed": selected["report"]["targets_passed"],
             "fitted_parameters": ["parent_bias", "leaf_bias"], "targets": copy.deepcopy(TARGETS),
             "fit_splits": ["val_known", "val_intra", "val_extra"],
             "input_record_count": input_count, "unique_image_count": len(rows),
             "duplicate_record_count": input_count - len(rows),
             "fit_image_sha256": sorted(_digest(r) for r in rows),
             "grid": {"parent_bias": pg.tolist(), "leaf_bias": lg.tolist(),
                      "sampled_point_count": selected["sampled_point_count"],
                      "sampled_feasible_count": selected["sampled_feasible_count"]},
             "grid_tradeoff": selected["points"]}
    report = evaluate_records(decode_records(rows, state, meta), meta)
    report["fit_completed"] = True
    report["sampled_feasible_count"] = selected["sampled_feasible_count"]
    report["individual_sampled_maxima"] = selected["individual_sampled_maxima"]
    report["infeasibility"] = {
        "no_feasible_sampled_point": selected["sampled_feasible_count"] == 0,
        "failed_at_selected_point": [name for name, passed in report["checks"].items() if not passed],
        "scope": "finite declared grid only; not a proof of continuous infeasibility",
        "selection_rule": "feasible first, then total normalized count deficit, mean of four metrics, known accuracy, smallest absolute biases",
    }
    state["validation_report"] = report
    state["source_loo"] = (source_loo(known, near, extra, meta, settings)
                           if settings.get("source_loo", True) else
                           {"available": False, "folds": [], "reason": "disabled in configuration"})
    evidence = [{"image_sha256": _digest(row), "split": row["split"], "status": row["status"],
                 "true_parent": row.get("true_parent"), "true_leaf": row.get("true_leaf"),
                 "log_probs": np.asarray(row["log_probs"], dtype=float).tolist()}
                for row in sorted(rows, key=_digest)]
    state["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
    binding = {"meta": meta, "hashes": state["fit_image_sha256"], "evidence_sha256": state["evidence_sha256"],
               "parent_bias": state["parent_bias"],
               "leaf_bias": state["leaf_bias"], "grid": state["grid"], "targets": TARGETS}
    state["calibration_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return state
