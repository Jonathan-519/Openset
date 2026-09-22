"""Empirical known-utility floor for child rejection; never a risk guarantee."""
import numpy as np


def protected_thresholds(out, rows, reference_predictions, root_predictions, meta, indices):
    """Per-parent constraints preserve reference micro AND macro known utility.

    Denominators include every known query in the true branch, including wrong
    routes/candidates. Unknown objective includes wrong-parent intra routes.
    Infeasible branches are explicitly reported, never relabelled as success.
    """
    n = len(rows)
    if len(reference_predictions) != n or len(root_predictions) != n:
        raise ValueError("Calibration inputs differ in length")
    ids = np.asarray(indices, dtype=int)
    if not len(ids) or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= n).any():
        raise ValueError("Invalid calibration partition")
    if any(not rows[i]["split"].startswith("val_") for i in ids):
        raise ValueError("Thresholds may only use validation records")
    held = np.zeros(n, bool); held[ids] = True
    status = np.asarray([r["status"] for r in rows])
    true_p = np.asarray([-1 if r["true_parent"] is None else r["true_parent"] for r in rows])
    true_l = np.asarray([-1 if r["true_leaf"] is None else r["true_leaf"] for r in rows])
    source = np.asarray([r["source"] for r in rows])
    root_ok = np.asarray([r["prediction_type"] != "global_unknown" for r in root_predictions])
    base_correct = np.asarray([r["prediction_type"] == "known" and r["candidate_leaf"] == r["true_leaf"]
                               and r["candidate_parent"] == r["true_parent"] for r in reference_predictions])
    scores = np.asarray(out["score"], dtype=float)
    if not np.isfinite(scores[ids]).all():
        raise ValueError("Nonfinite calibration score")
    pp, pl = np.asarray(out["parent"]), np.asarray(out["leaf"])
    if not np.array_equal(pp, [r["candidate_parent"] for r in root_predictions]):
        raise ValueError("Changed frozen root routing")
    thresholds, reports = {}, {}
    for p in range(len(meta["parent_names"])):
        positive = held & (status == "known") & (true_p == p)
        negative = held & (status == "intra") & (pp == p) & root_ok
        routed = held & (pp == p) & root_ok
        values = scores[routed]
        candidates = np.unique(np.r_[values, np.nextafter(scores[ids].min(), -np.inf),
                                      np.nextafter(scores[ids].max(), np.inf)])
        groups = sorted(set(true_l[positive].tolist()))
        sources = sorted(set(source[negative].tolist()))
        def utility(correct):
            return (float(correct[positive].mean()) if positive.any() else 0.,
                    float(np.mean([correct[positive & (true_l == c)].mean() for c in groups])) if groups else 0.)
        baseline_micro, baseline_macro = utility(base_correct)
        best = None
        for tau in candidates:
            accept = root_ok & (scores >= tau)
            correct = accept & (pl == true_l) & (pp == true_p)
            micro, macro = utility(correct)
            feasible = bool(positive.any() and micro + 1e-12 >= baseline_micro and macro + 1e-12 >= baseline_macro)
            rejection = float(np.mean([(~accept)[negative & (source == s)].mean() for s in sources])) if sources else 0.
            # Within feasible points prioritize unknown rejection. If infeasible,
            # return maximal known utility with a failure flag, not silent fallback.
            key = ((1, rejection, macro, micro, -float(tau)) if feasible
                   else (0, macro, micro, rejection, -float(tau)))
            if best is None or key > best[0]:
                best = (key, float(tau), feasible, micro, macro, rejection)
        thresholds[str(p)] = best[1]
        reports[str(p)] = {"known_count": int(positive.sum()), "routed_intra_count": int(negative.sum()),
                           "intra_sources": len(sources), "reference_micro": baseline_micro,
                           "reference_macro": baseline_macro, "achieved_micro": best[3],
                           "achieved_macro": best[4], "source_macro_rejection": best[5],
                           "feasible": best[2], "unknown_evidence_available": bool(sources)}
    return {"thresholds": thresholds, "parents": reports,
            "all_known_floors_feasible": all(v["feasible"] for v in reports.values()),
            "claim": "Empirical held-validation utility only; no test or unseen-species guarantee"}
