"""Reference-bound cache, independent fitting, calibration and immutable TEST."""
import copy
import csv
from pathlib import Path
import time
from types import SimpleNamespace
from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_refine.importer import inspect_reference
from . import protocol,features,training,calibration


def _checked(suite):
    from . import runner
    suite=Path(suite).resolve();cfg=protocol.validate_config(protocol.read_json(runner._regular(suite/"config.json")))
    binding=protocol.read_json(runner._regular(suite/"source_binding.json"))
    info=inspect_reference(binding["directory"])
    runner._verify_snapshot(suite,cfg,info)
    return suite,cfg,info


def _header(cfg,info,stage,arm=None):
    return dict(schema_version=protocol.SCHEMA_VERSION,signature=protocol.signature(cfg,info["binding"]),
        source_binding=info["binding"],meta=info["meta"],stage=stage,arm_id=arm,
        test_used_for_fitting=False,unknown_images_used_for_gradients=False)


def _claim(path):
    path.mkdir(parents=True,exist_ok=False)
    return path


def _complete(output,receipt,names):
    receipt["artifacts"]={key:dict(path=name,sha256=protocol.file_hash(output/name)) for key,name in names.items()}
    protocol.write_json(output/"completed.json",receipt)
    return receipt


def prepare_cache(suite,stage,device):
    from . import reporting
    suite,cfg,info=_checked(suite)
    if stage=="test":reporting.freeze_dev_selection(suite)
    groups,audit=features.source_rows(info,stage)
    image_audit=features.audit_images(groups)
    if not image_audit["valid"]:raise ValueError("Missing/changed source images: "+str(image_audit["problems"]))
    output=_claim(suite/"cache"/stage);started=time.perf_counter()
    cache=features.collect(info,groups,device)
    cache.update(signature=protocol.signature(cfg,info["binding"]),stage=stage,audit=audit)
    support._save_torch(output/"features.pth",cache)
    protocol.write_json(output/"raw_image_audit.json",image_audit)
    receipt=dict(_header(cfg,info,stage),audit=audit,feature_summary=features.summary(cache),seconds=time.perf_counter()-started)
    if stage=="test":receipt["dev_selection_sha256"]=protocol.file_hash(suite/"dev_selection.json")
    return _complete(output,receipt,dict(features="features.pth",image_audit="raw_image_audit.json"))


def _load_cache(suite,stage,cfg,info):
    from . import runner
    snapshot=runner._verify_snapshot(suite,cfg,info)
    receipt=runner._verify_stage(suite,None,"cache_"+stage,snapshot)
    if stage=="test":
        from . import reporting
        reporting.freeze_dev_selection(suite)
        if receipt["dev_selection_sha256"]!=protocol.file_hash(suite/"dev_selection.json"):
            raise ValueError("TEST cache DEV freeze changed")
    cache=support._load_torch(suite/"cache"/stage/"features.pth")
    if cache["source_binding"]!=info["binding"] or cache["signature"]!=snapshot["signature"] or cache["stage"]!=stage:
        raise ValueError("Feature cache identity changed")
    return cache,receipt


def fit_arm(suite,arm_id,device):
    suite,cfg,info=_checked(suite);arm=next(a for a in cfg["arms"] if a["id"]==arm_id)
    cache,cached=_load_cache(suite,"train",cfg,info)
    output=_claim(suite/"arms"/arm_id/"training")
    if arm["kind"]=="reference":
        report=dict(training_execution="exact_C00_inherited",optimizer_steps=0,frozen_encoder_updated=False,
                    old_C00_checkpoint_modified=False,parameter_delta_l2=0.,gradient_splits=[],test_used_for_fitting=False)
        payload=dict(reference=True,report=report)
    else:
        payload,report=training.fit(cache,info["meta"],arm,cfg,device)
    payload.update(signature=protocol.signature(cfg,info["binding"]),source_binding=info["binding"],
                   arm=arm,train_cache_sha256=cached["artifacts"]["features"]["sha256"])
    support._save_torch(output/"model.pth",payload)
    protocol.write_json(output/"training_report.json",report)
    return _complete(output,dict(_header(cfg,info,"training",arm_id),optimizer_steps=report["optimizer_steps"],
        train_cache_sha256=payload["train_cache_sha256"],training_report=report),dict(model="model.pth",report="training_report.json"))


def _load_model(suite,arm,cfg,info):
    from . import runner
    snapshot=runner._verify_snapshot(suite,cfg,info)
    receipt=runner._verify_stage(suite,arm["id"],"training",snapshot)
    cache_receipt=runner._verify_stage(suite,None,"cache_train",snapshot)
    payload=support._load_torch(suite/"arms"/arm["id"]/"training/model.pth")
    if (payload["signature"]!=snapshot["signature"] or payload["source_binding"]!=info["binding"]
            or payload["arm"]!=arm or payload["train_cache_sha256"]!=cache_receipt["artifacts"]["features"]["sha256"]):
        raise ValueError("Trained model source/cache/arm identity changed")
    return payload,receipt


def _rows(groups):
    return [row for value in groups.values() for row in value]


def _species(output,predictions,meta):
    rows=base.unique_records(predictions);groups={}
    for row in rows:
        species=meta["leaf_names"][row["true_leaf"]] if row["status"]=="known" else row["source"]
        groups.setdefault((row["status"],species),[]).append(row)
    values=[]
    for (status,name),group in sorted(groups.items()):
        report=base._group_report(group,status)
        wrong_parent=sum(x["prediction_type"]=="intra_unknown" and x["parent"]!=x.get("true_parent") for x in group) if status=="intra" else 0
        false_leaf=sum(x["prediction_type"]=="known" and (status!="known" or x["leaf"]!=x["true_leaf"]) for x in group)
        values.append(dict(status=status,species=name,sample_count=len(group),correct_count=report["correct_count"],
            correct_terminal_rate=report["correct_rate"],root_count=sum(x["prediction_type"]=="global_unknown" for x in group),
            parent_count=sum(x["prediction_type"]=="intra_unknown" for x in group),wrong_parent_count=wrong_parent,
            false_leaf_count=false_leaf,evidence_status="insufficient_evidence" if len(group)<5 else "observed"))
    with (output/"per_species.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def _export(output,rows,router,arm,info,oof,diagnostics,receipt):
    pred=membership.decode_records(rows,router,info["meta"]) if arm["kind"]=="reference" else calibration.decode(rows,router,info["meta"])
    seen=set()
    for row in pred:
        row["evaluation_weight"]=int(row["image_sha256"] not in seen);seen.add(row["image_sha256"])
    summary=base.evaluate_records(pred,info["meta"])
    before=membership.decode_records(rows,info["router"],info["meta"])
    paired=calibration.paired_audit(before,pred,info["meta"])
    report=dict(summary,paired_to_C00=paired,calibration_diagnostics=diagnostics,
        crossfit_passed=oof["passed"],crossfit_complete=oof["complete"],
        calibration_gate_is_execution_gate=False,test_allowed_after_failed_gates=True,
        test_used_for_fitting=False,confirmatory_validation=False,
        validation_scope="exploratory_C00_joint_on_reused_DEV_and_TEST")
    protocol.write_records(output/"scores.jsonl",rows);protocol.write_records(output/"predictions.jsonl",pred)
    protocol.write_json(output/"summary.json",summary);protocol.write_json(output/"report.json",report)
    protocol.write_json(output/"router.json",router);protocol.write_json(output/"crossfit_audit.json",oof)
    _species(output,pred,info["meta"])
    receipt.update(targets_passed=summary["targets_passed"],summary=summary,crossfit_passed=oof["passed"],
        calibration_gate_is_execution_gate=False,test_allowed_after_failed_gates=True)
    return _complete(output,receipt,dict(scores="scores.jsonl",predictions="predictions.jsonl",summary="summary.json",
        report="report.json",router="router.json",crossfit="crossfit_audit.json",species="per_species.csv"))


def calibrate_arm(suite,arm_id,device):
    suite,cfg,info=_checked(suite);arm=next(a for a in cfg["arms"] if a["id"]==arm_id)
    cache,cached=_load_cache(suite,"development",cfg,info);payload,trained=_load_model(suite,arm,cfg,info)
    groups={k:v["records"] for k,v in cache["groups"].items()} if arm["kind"]=="reference" else training.score(cache,payload,device)
    rows=_rows(groups)
    # Reuse only verified C00 fit-fold reference routers; no candidate scores are reused.
    original=suite/"arms/A00_reference/calibration/crossfit_audit.json"
    if arm["kind"]!="reference" and original.exists():
        from . import runner
        runner._verify_stage(suite,"A00_reference","calibration",runner._verify_snapshot(suite,cfg,info))
        folds=protocol.read_json(original)["reference_folds"]
    else:
        folds=calibration.reference_folds(rows,info["meta"],cfg["calibration"],info["config"]["calibration"])
    oof=calibration.crossfit(rows,info["meta"],cfg["calibration"],folds,arm["kind"]=="reference")
    if arm["kind"]=="reference":
        router=copy.deepcopy(info["router"]);diagnostics=dict(source_reproduction=cache["source_reproduction"])
    else:
        router,diagnostics=calibration.fit(rows,info["meta"],cfg["calibration"],info["router"])
        raw=calibration.decode(rows,calibration._state(info["meta"],0.,0.,[]),info["meta"])
        diagnostics["uncalibrated_joint_argmax"]=base.evaluate_records(raw,info["meta"])
    receipt=dict(_header(cfg,info,"calibration",arm_id),model_sha256=trained["artifacts"]["model"]["sha256"],
        training_receipt_sha256=protocol.file_hash(suite/"arms"/arm_id/"training/completed.json"),
        cache_sha256=cached["artifacts"]["features"]["sha256"])
    return _export(_claim(suite/"arms"/arm_id/"calibration"),rows,router,arm,info,oof,diagnostics,receipt)


def test_arm(suite,arm_id,device):
    from . import runner,reporting
    suite,cfg,info=_checked(suite);reporting.freeze_dev_selection(suite)
    arm=next(a for a in cfg["arms"] if a["id"]==arm_id)
    snapshot=runner._verify_snapshot(suite,cfg,info)
    calibrated=runner._verify_stage(suite,arm_id,"calibration",snapshot)
    payload,trained=_load_model(suite,arm,cfg,info)
    if calibrated["model_sha256"]!=trained["artifacts"]["model"]["sha256"]:
        raise ValueError("TEST model differs from calibrated weights")
    cache,cached=_load_cache(suite,"test",cfg,info)
    groups={k:v["records"] for k,v in cache["groups"].items()} if arm["kind"]=="reference" else training.score(cache,payload,device)
    directory=suite/"arms"/arm_id/"calibration"
    router=protocol.read_json(directory/"router.json");oof=protocol.read_json(directory/"crossfit_audit.json")
    receipt=dict(_header(cfg,info,"test",arm_id),model_sha256=trained["artifacts"]["model"]["sha256"],
        calibration_receipt_sha256=protocol.file_hash(directory/"completed.json"),
        router_sha256=protocol.file_hash(directory/"router.json"),cache_sha256=cached["artifacts"]["features"]["sha256"],
        dev_selection_sha256=protocol.file_hash(suite/"dev_selection.json"))
    return _export(_claim(suite/"arms"/arm_id/"test"),_rows(groups),router,arm,info,oof,
        dict(crossfit_origin="frozen_DEV_only; no TEST fitting"),receipt)
