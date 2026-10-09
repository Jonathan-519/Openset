"""Descriptive score bounds, kept separate from executable calibration rules."""
import numpy as np

from taxosafe_support import calibration as base
from taxosafe_discovery.calibration import _boundaries


def _auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0) + .5 * (differences == 0)).mean())


def score_diagnostics(records):
    rows = base.unique_records(records)
    if not rows or any("morphology" not in row for row in rows):
        return None
    score = np.array([r["morphology"]["root_score"] for r in rows])
    known, near, extra = [np.array([r["status"] == s for r in rows]) for s in base.STATUSES]
    correct_known = known & np.array([r["morphology"]["anchor_leaf"] == r.get("true_leaf") for r in rows])
    correct_near = near & np.array([r["morphology"]["anchor_parent"] == r.get("true_parent") for r in rows])
    distributions = {}
    for name, mask in (("known", known), ("near", near), ("extra", extra)):
        distributions[name] = dict(count=int(mask.sum()), quantiles=np.quantile(score[mask],[0,.1,.5,.9,1]).tolist()) if mask.any() else None
    result = dict(root_score_distribution=distributions,
                  near_vs_extra_root_auroc=_auc(score[near],score[extra]),
                  known_plus_near_vs_extra_root_auroc=_auc(score[known|near],score[extra]),
                  root_candidate_ceiling=dict(known=int(correct_known.sum()),near=int(correct_near.sum())),
                  diagnostic_only=True, changes_executable_router=False)
    if any(not r["split"].startswith("val_") for r in rows):
        result["necessary_DEV_bound"] = None
        return result
    required = dict(known=9*int(known.sum())//10+1,near=(85*int(near.sum())+99)//100,
                    extra=9*int(extra.sum())//10+1)
    threshold=_boundaries(score)
    passed=score[None,:]>=threshold[:,None]
    k=(passed&correct_known).sum(1);n=(passed&correct_near).sum(1);e=((~passed)&extra).sum(1)
    joint=(k>=required["known"])&(n>=required["near"])
    konly=k>=required["known"]
    maximum=int(e[joint].max()) if joint.any() else None
    result["necessary_DEV_bound"]=dict(required=required,
        maximum_extra_rejected_protecting_known_and_near=maximum,
        maximum_extra_rejected_protecting_known_only=int(e[konly].max()) if konly.any() else None,
        necessary_condition_met=bool(maximum is not None and maximum>=required["extra"]),
        interpretation="Necessary empirical root condition, not four-gate qualification and not a replacement for the 92% staged root buffer")
    return result
