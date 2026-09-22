"""Exact parent/leaf TRAIN conformal components for the v9 safety interlock."""
import numpy as np

from taxosafe_regime import core as regime

FEATURES = regime.FEATURES
VARIANTS = regime.VARIANTS
prepare, subset, evidence = regime.prepare, regime.subset, regime.evidence
fit_all = regime.fit_all


def pvalue_components(values, parents, leaves, state, variant):
    """Return exact finite-sample empirical p-values at parent and leaf levels.

    Scores increase with compatibility.  Each component is calibrated only on
    the known TRAIN calibration partition.  The two components enter separate
    veto and rescue regions of the downstream hysteresis rule.
    """
    values = np.asarray(values, dtype=float)
    parents = np.asarray(parents, dtype=int)
    leaves = np.asarray(leaves, dtype=int)
    if values.shape != parents.shape or values.shape != leaves.shape:
        raise ValueError("Score, parent and leaf arrays must have equal shape")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid verifier scores")
    groups = state["variants"][variant]
    parent_p = np.empty(len(values), dtype=float)
    leaf_p = np.empty(len(values), dtype=float)
    for i, (value, parent, leaf) in enumerate(zip(values, parents, leaves)):
        parent_scores = np.asarray(groups["parents"][str(int(parent))]["scores"], dtype=float)
        leaf_scores = np.asarray(groups["leaves"][str(int(leaf))]["scores"], dtype=float)
        if not len(parent_scores) or not len(leaf_scores):
            raise ValueError("Missing parent/leaf TRAIN calibration group")
        parent_p[i] = (1.0 + np.searchsorted(parent_scores, value, side="right")) / (len(parent_scores) + 1.0)
        leaf_p[i] = (1.0 + np.searchsorted(leaf_scores, value, side="right")) / (len(leaf_scores) + 1.0)
    return parent_p, leaf_p
