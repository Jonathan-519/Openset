"""Leaf-parent partially pooled calibration over the v7 dual evidence score."""
import numpy as np
from taxosafe_dual import core as dual
from taxosafe_witness import core as witness

FEATURES = dual.FEATURES
VARIANTS = dual.VARIANTS
prepare, subset, evidence = dual.prepare, dual.subset, dual.evidence
training_world = dual.training_world

def build_calibration(models, data, rows, meta, support, calibration, settings):
    y = np.asarray([r["true_leaf"] for r in rows], int)
    mapping = np.asarray(meta["leaf_to_parent"], int)
    e = evidence(subset(data, calibration), subset(data, support), y[support],
                 mapping, mapping[y[calibration]], settings, candidates=y[calibration])
    variants = {}
    for variant, model in models.items():
        values = witness.score(model, e); parents = {}; leaves = {}
        for parent in range(len(meta["parent_names"])):
            ids = np.flatnonzero(mapping[y[calibration]] == parent)
            if not len(ids): raise ValueError("Every parent needs TRAIN calibration images")
            parents[str(parent)] = {"count": int(len(ids)),
                                    "scores": sorted(float(values[i]) for i in ids)}
        for leaf in range(len(meta["leaf_names"])):
            ids = np.flatnonzero(y[calibration] == leaf)
            if not len(ids): raise ValueError("Every leaf needs TRAIN calibration images")
            leaves[str(leaf)] = {"count": int(len(ids)),
                                 "scores": sorted(float(values[i]) for i in ids)}
        variants[variant] = {"parents": parents, "leaves": leaves}
    return {"schema": 1, "method": "leaf_parent_partial_pooling",
            "alpha": float(settings["conformal_alpha"]),
            "shrinkage": float(settings["hierarchical_shrinkage"]),
            "acceptance": "pooled_p_value > alpha on multi-leaf branches; legacy coverage on singleton branches",
            "unknown_labels_used": False, "validation_used": False, "test_used": False,
            "calibration_query_ids": calibration.tolist(),
            "calibration_image_hashes": [rows[i]["image_sha256"] for i in calibration],
            "variants": variants}

def pvalues(values, parents, leaves, state, variant):
    """Candidate-leaf empirical CDF shrunk toward its parent CDF."""
    values, parents, leaves = np.asarray(values,float),np.asarray(parents,int),np.asarray(leaves,int)
    if values.shape != parents.shape or values.shape != leaves.shape or not np.isfinite(values).all():
        raise ValueError("Invalid score arrays")
    lam=float(state["shrinkage"])
    if lam < 0: raise ValueError("Nonnegative hierarchical shrinkage required")
    groups=state["variants"][variant]; out=np.empty(len(values),float)
    for i,(value,parent,leaf) in enumerate(zip(values,parents,leaves)):
        lr=np.asarray(groups["leaves"][str(int(leaf))]["scores"],float)
        pr=np.asarray(groups["parents"][str(int(parent))]["scores"],float)
        if not len(lr) or not len(pr): raise ValueError("Missing hierarchical calibration group")
        lc=float(np.searchsorted(lr,value,side="right"))
        parent_cdf=(1.+np.searchsorted(pr,value,side="right"))/(len(pr)+1.)
        out[i]=(1.+lc+lam*parent_cdf)/(len(lr)+1.+lam)
    return out

def fit_all(data, rows, meta, settings, seed):
    models, _, report = dual.fit_all(data, rows, meta, settings, seed)
    support=np.asarray(report["split"]["support"],int)
    calibration=np.asarray(report["split"]["calibration"],int)
    state=build_calibration(models, prepare(data,settings), rows, meta, support, calibration, settings)
    report["calibration_method"]="candidate-leaf empirical CDF partially pooled toward parent"
    report["hierarchical_shrinkage"]=float(settings["hierarchical_shrinkage"])
    report["singleton_policy"]="inherit frozen v4 coverage because child novelty is unidentifiable"
    report["guarantee_scope"]="TRAIN-only hierarchical calibration; partial pooling is empirical, not an exact finite-sample conformal guarantee"
    return models,state,report
