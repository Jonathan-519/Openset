"""DEV-only operating-point calibration of learned multiclass terminal logits.

Two preregistered type biases adjust the joint network, not the old global
root score. All parents and leaves compete simultaneously; TEST never fits.
"""
import copy
import numpy as np
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_routealign.calibration import _folds,paired_audit
from .protocol import object_hash

SCHEMA="reference_joint_router_v1"


def _state(meta,parent_bias,root_bias,hashes):
    value=dict(schema_version=SCHEMA,meta=meta,parent_bias=float(parent_bias),root_bias=float(root_bias),
        fit_image_sha256=sorted(hashes),fit_splits=["val_known","val_intra","val_extra"],
        test_used_for_fitting=False,decoder="joint_terminal_argmax")
    value["state_sha256"]=object_hash(value)
    return value


def decode(records,router,meta):
    if router.get("schema_version")!=SCHEMA or router.get("meta")!=meta:
        raise ValueError("Joint router schema/taxonomy changed")
    unsigned={k:v for k,v in router.items() if k!="state_sha256"}
    if router.get("state_sha256")!=object_hash(unsigned) or router.get("test_used_for_fitting") is not False:
        raise ValueError("Joint router identity changed")
    c,p=len(meta["leaf_names"]),len(meta["parent_names"])
    logits=np.asarray([r["joint_logits"] for r in records],dtype=np.float64)
    if logits.shape!=(len(records),c+p+1) or not np.isfinite(logits).all():
        raise ValueError("Invalid learned terminal scores")
    scores=logits.copy();scores[:,c:c+p]+=router["parent_bias"];scores[:,-1]+=router["root_bias"]
    result=[]
    for row,values in zip(records,scores):
        terminal=int(values.argmax());candidate=int(values[:c].argmax())
        if terminal<c:
            leaf=terminal;parent=int(meta["leaf_to_parent"][leaf]);kind="known";node=1+p+leaf
        elif terminal<c+p:
            leaf=None;parent=terminal-c;kind="intra_unknown";node=1+parent
        else:
            leaf=parent=None;kind="global_unknown";node=0
        result.append(dict(row,prediction_type=kind,parent=parent,leaf=leaf,output_node=node,
            candidate_leaf=candidate,candidate_parent=int(meta["leaf_to_parent"][candidate]),
            candidate_leaf_name=meta["leaf_names"][candidate],
            candidate_parent_name=meta["parent_names"][meta["leaf_to_parent"][candidate]],
            root_knownness_score=float(values[:-1].max()-values[-1]),
            local_knownness_score=float(values[:c].max()-values[c:].max()),
            decoder="joint_terminal_argmax",score_note="learned multiclass logits; not probabilities"))
    return result


def macro(report,key):
    values=[r["correct_rate"] for r in report[key].values() if r["sample_count"]]
    return sum(values)/len(values) if values else 0.0


def fit(rows,meta,settings,reference_router):
    rows=base.unique_records(rows)
    if any(not r["split"].startswith("val_") for r in rows) or set(r["status"] for r in rows)!=set(base.STATUSES):
        raise ValueError("Joint calibration requires only complete DEV statuses")
    baseline=membership.decode_records(rows,reference_router,meta)
    original=base.evaluate_records(baseline,meta)
    best=None;grid=[]
    for parent in settings["bias_grid"]:
        for root in settings["bias_grid"]:
            state=_state(meta,parent,root,[r["image_sha256"] for r in rows])
            pred=decode(rows,state,meta);report=base.evaluate_records(pred,meta)
            preserved=report["counts"]["known_correct"]>=original["counts"]["known_correct"]
            macro_preserved=macro(report,"per_known_leaf")+1e-12>=macro(original,"per_known_leaf")
            metrics=[v or 0. for v in report["metrics"].values()]
            ratios=[metrics[0]/.9,metrics[1]/.85,metrics[2]/.9,metrics[3]/.9]
            # Scientific feasibility first; otherwise preserve the original C00 known task.
            rank=(int(report["targets_passed"] and preserved and macro_preserved),int(preserved and macro_preserved),
                  int(report["checks"]["known_end_to_end_leaf_accuracy"]),min(ratios),sum(metrics),
                  report["counts"]["known_correct"],-abs(parent)-abs(root),-parent,-root)
            grid.append(dict(parent_bias=parent,root_bias=root,metrics=report["metrics"],counts=report["counts"],
                             targets_passed=report["targets_passed"],known_count_preserved=preserved,known_macro_preserved=macro_preserved))
            if best is None or rank>best[0]:best=(rank,state,report,pred,preserved,macro_preserved)
    _,router,report,pred,preserved,macro_preserved=best
    diagnostics=dict(operating_point_grid=grid,known_count_preserved=preserved,known_macro_preserved=macro_preserved,
                     paired_to_C00=paired_audit(baseline,pred,meta),grid_selected_on="DEV_only",
                     architecture="learned joint leaf/parent-unknown/root competition")
    return router,diagnostics


def reference_folds(rows,meta,settings,reference_settings):
    rows=base.unique_records(rows);by_hash={r["image_sha256"]:r for r in rows};result=[]
    for fold in _folds(rows,settings):
        fitted=[by_hash[h] for h in fold["fit_image_sha256"]];held=[by_hash[h] for h in fold["held_image_sha256"]]
        groups=[[r for r in fitted if r["status"]==s] for s in base.STATUSES]
        if not held or any(not g for g in groups):
            result.append(dict(fold,status="not_evaluable"));continue
        router=membership.calibrate(*groups,meta,dict(reference_settings,source_loo=False))
        result.append(dict(fold,status="completed",reference_router=router,
                           reference_predictions=membership.decode_records(held,router,meta)))
    return result


def crossfit(rows,meta,settings,folds,reference=False):
    rows=base.unique_records(rows);by_hash={r["image_sha256"]:r for r in rows}
    before=[];after=[];audits=[]
    for fold in folds:
        audit={k:v for k,v in fold.items() if k not in ("reference_router","reference_predictions")}
        if fold["status"]!="completed":audits.append(audit);continue
        fit_rows=[by_hash[h] for h in fold["fit_image_sha256"]]
        held=[by_hash[h] for h in fold["held_image_sha256"]]
        if reference:
            predictions=fold["reference_predictions"]
        else:
            router,_=fit(fit_rows,meta,settings,fold["reference_router"])
            predictions=decode(held,router,meta)
            audit["joint_router"]=router
        before.extend(fold["reference_predictions"]);after.extend(predictions);audits.append(audit)
    complete=len(after)==len(rows) and len({r["image_sha256"] for r in after})==len(rows) and all(f["status"]=="completed" for f in audits)
    report=base.evaluate_records(after,meta) if after else None
    ref=base.evaluate_records(before,meta) if before else None
    preserve=bool(report and ref and report["counts"]["known_correct"]>=ref["counts"]["known_correct"] and
                  macro(report,"per_known_leaf")+1e-12>=macro(ref,"per_known_leaf"))
    return dict(schema_version="reference_joint_crossfit_v1",complete=complete,
        passed=bool(complete and report["targets_passed"] and preserve),folds=audits,report=report,
        known_count_and_macro_preserved=preserve,held_image_count=len(after),
        paired_to_C00=paired_audit(before,after,meta) if complete else None,
        reference_predictions=before,predictions=after,reference_folds=folds if reference else None,
        model_refitted_in_folds=False,validation_scope="conditional calibration-only OOF on fixed TRAIN model",
        independent_model_level_validation=False,confirmatory_validation=False,held_data_used_for_biases=False,
        test_used_for_fitting=False,output_used_for_threshold_selection=False)
