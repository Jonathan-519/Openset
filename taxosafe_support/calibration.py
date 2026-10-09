"""Reference membership routing and shared unique-content evaluation contracts."""
from collections import Counter, defaultdict
import copy

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
SELECTION_POLICIES = ("balanced", "known_first")


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
            if "support_evidence" in old or "support_evidence" in row:
                if "support_evidence" not in old or "support_evidence" not in row:
                    raise ValueError("Same content has missing raw support evidence: " + key)
                left, right = old["support_evidence"], row["support_evidence"]
                if (not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys()
                        or any(not np.array_equal(left[k], right[k]) for k in left)):
                    raise ValueError("Same content has inconsistent raw support evidence: " + key)
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


def _selection_policy(settings):
    policy = settings.get("policy", "balanced")
    if policy not in SELECTION_POLICIES:
        raise ValueError("Calibration policy must be balanced or known_first")
    return policy


def decode_records(records, state, meta):
    from .membership_calibration import decode_records as decode_membership
    return decode_membership(records, state, meta)


apply_router = decode_records


def calibrate(known, near, extra, meta, settings=None):
    from .membership_calibration import calibrate as calibrate_membership
    settings = dict(settings or {})
    if settings.get("decoder", "membership") != "membership":
        raise ValueError("Only the reference membership decoder is retained")
    return calibrate_membership(known, near, extra, meta, settings)


def source_loo(known, near, extra, meta, settings=None):
    from .membership_calibration import source_loo as membership_loo
    return membership_loo(known, near, extra, meta, settings)


def fixed_score_feasibility(known, near, extra, meta):
    from .membership_calibration import fixed_score_feasibility as membership_feasibility
    return membership_feasibility(known, near, extra, meta)
