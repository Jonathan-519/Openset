"""Known-only calibration with a separate child audit partition.

No unknown validation labels or scores select the model or thresholds.
Post-rejection accuracy on future data is NOT guaranteed by fixed class labels.
"""
import numpy as np


def partition(rows, held_indices, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be inside (0,1)")
    ids = np.asarray(held_indices, int)
    if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= len(rows)).any():
        raise ValueError("Invalid original held partition")
    if any(not rows[i]["split"].startswith("val_") for i in ids):
        raise ValueError("Calibration partition must be validation")
    rng = np.random.RandomState(seed)
    cal = []
    for leaf in sorted({rows[i]["true_leaf"] for i in ids if rows[i]["status"] == "known"}):
        group = np.asarray(sorted([i for i in ids if rows[i]["status"] == "known" and rows[i]["true_leaf"] == leaf],
                                  key=lambda i: rows[i]["image_sha256"]))
        group = rng.permutation(group)
        n = max(1, int(np.ceil(len(group) * fraction)))
        cal.extend(group[:n].tolist())
    audit = sorted(set(ids.tolist()) - set(cal))
    if not cal or not audit:
        raise ValueError("Need nonempty calibration and audit partitions")
    if not any(rows[i]["status"] == "known" for i in audit):
        raise ValueError("Need known images disjoint from child threshold calibration for audit")
    return sorted(cal), audit


def _arrays(scores, rows, root, indices):
    ids = np.asarray(indices, int)
    if any(rows[i]["status"] != "known" or rows[i]["split"] != "val_known" for i in ids):
        raise ValueError("Threshold fitting accepts known validation ONLY")
    s = np.asarray(scores, float)
    if len(s) != len(rows) or len(root) != len(rows) or not np.isfinite(s[ids]).all():
        raise ValueError("Invalid calibration arrays")
    y = np.asarray([-1 if r["true_leaf"] is None else r["true_leaf"] for r in rows])
    p = np.asarray([-1 if r["true_parent"] is None else r["true_parent"] for r in rows])
    route = np.asarray([r["candidate_parent"] for r in root])
    leaf = np.asarray([r["candidate_leaf"] for r in root])
    correct = (leaf == y) & (route == p) & np.asarray([r["prediction_type"] != "global_unknown" for r in root])
    return ids, s, y, p, route, correct


def reference_thresholds(scores, rows, root, indices, parent_count, retention):
    if not 0 < retention <= 1:
        raise ValueError("Known retention must be in (0,1]")
    ids, s, y, p, route, correct = _arrays(scores, rows, root, indices)
    usable = ids[correct[ids]]
    if not len(usable):
        raise ValueError("No correctly classified known calibration images")
    def quantile(values):
        count = max(1, int(np.ceil(len(values) * retention)))
        return float(np.sort(values)[-count])
    pooled = quantile(s[usable])
    thresholds, report = {}, {}
    for parent in range(parent_count):
        good = usable[p[usable] == parent]
        thresholds[str(parent)] = quantile(s[good]) if len(good) else pooled
        report[str(parent)] = {"correct_calibration_count": len(good),
                               "fallback": "pooled_known" if not len(good) else None}
    return thresholds, report


def matched_thresholds(scores, rows, root, indices, reference_accept, parent_count, retention):
    ids, s, y, p, route, correct = _arrays(scores, rows, root, indices)
    pooled, fallback_report = reference_thresholds(s, rows, root, indices, parent_count, retention)
    reference_accept = np.asarray(reference_accept, bool)
    thresholds, reports = {}, {}
    for parent in range(parent_count):
        group = ids[p[ids] == parent]
        leaves = sorted(set(y[group].tolist()))
        def utility(accepted):
            c = accepted & correct
            return (float(c[group].mean()) if len(group) else 0.,
                    float(np.mean([c[group[y[group] == leaf]].mean() for leaf in leaves])) if leaves else 0.)
        target = utility(reference_accept)
        if not len(group) or not correct[group].any():
            thresholds[str(parent)] = pooled[str(parent)]
            reports[str(parent)] = {"known_count": len(group), "feasible": False,
                                    "fallback": "pooled_known", "reference": target}
            continue
        values = s[group[correct[group]]]
        candidates = np.unique(np.r_[values, np.nextafter(values.max(), np.inf)])[::-1]
        selected = None
        for tau in candidates:
            u = utility(s >= tau)
            if u[0] + 1e-12 >= target[0] and u[1] + 1e-12 >= target[1]:
                selected = (float(tau), u)
                break
        if selected is None:
            raise AssertionError("Fixed classifier must permit matching its reference known utility")
        thresholds[str(parent)] = selected[0]
        reports[str(parent)] = {"known_count": len(group), "reference": target, "achieved": selected[1],
                                "feasible": True, "fallback": None}
    return {"thresholds": thresholds, "parents": reports, "unknowns_used": False,
            "claim": "Known calibration utility constraint; not a guarantee on unseen data"}
