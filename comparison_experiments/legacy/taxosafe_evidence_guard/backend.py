"""Complete C00 cache, audited TRAIN exposure and immutable DEV/TEST lifecycle."""
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
        test_used_for_fitting=False,unknown_images_used_for_gradients=bool(stage in ("training", "calibration", "test") and
            arm and next(item for item in cfg["arms"] if item["id"] == arm)["use_oe"]),
        expanded_TRAIN_protocol=True)


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
    groups,audit=features.source_rows(info,stage,cfg)
    image_audit=features.audit_images(groups)
    if not image_audit["valid"]:raise ValueError("Missing/changed source images: "+str(image_audit["problems"]))
    if stage != "train":
        traincache,_ = _load_cache(suite,"train",cfg,info)
        forbidden = {row["image_sha256"] for group in traincache["groups"].values() for row in group["records"]}
        if any(row["image_sha256"] in forbidden for values in groups.values() for row in values):
            raise ValueError("Added TRAIN known/unknown overlaps the actual " + stage + " images")
        if stage == "test":
            from taxosafe_dcbs.protocol import normalized_name
            train_sources = {normalized_name(row["source"]) for name,group in traincache["groups"].items()
                             if name != "train" for row in group["records"]}
            if any(row["status"] != "known" and normalized_name(row["source"]) in train_sources
                   for values in groups.values() for row in values):
                raise ValueError("Added unknown TRAIN source appears in actual TEST")
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
    elif arm["kind"] == "reuse":
        source_arm = next(item for item in cfg["arms"] if item["id"] == arm["weight_source"])
        payload,source_receipt = _load_model(suite,source_arm,cfg,info)
        payload = copy.deepcopy(payload)
        source_report = payload["report"]
        report = dict(copy.deepcopy(source_report), training_execution="reuse_verified_R04_weights",
            optimizer_steps=0, source_optimizer_steps=source_report["optimizer_steps"],
            weight_source=source_arm["id"], reused_model_sha256=source_receipt["artifacts"]["model"]["sha256"])
        payload["report"] = report
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


def _export(output,rows,router,arm,info,oof,diagnostics,receipt,fold_scores=None):
    pred=calibration.decode(rows,router,info["meta"])
    seen=set()
    for row in pred:
        row["evaluation_weight"]=int(row["image_sha256"] not in seen);seen.add(row["image_sha256"])
    summary=base.evaluate_records(pred,info["meta"])
    before=membership.decode_records(calibration.reference_rows(rows),info["router"],info["meta"])
    paired=calibration.paired_audit(before,pred,info["meta"])
    report=dict(summary,paired_to_C00=paired,calibration_diagnostics=diagnostics,
        crossfit_passed=oof["passed"],crossfit_complete=oof["complete"],
        calibration_gate_is_execution_gate=False,test_allowed_after_failed_gates=True,
        test_used_for_fitting=False,confirmatory_validation=False,
        validation_scope="expanded_TRAIN_OE; conditional_adapter_source_OOF; reused_DEV_TEST",
        unknown_images_used_for_gradients=bool(arm["use_oe"]),
        source_training_expanded=bool(arm["use_oe"]))
    protocol.write_records(output/"scores.jsonl",rows);protocol.write_records(output/"predictions.jsonl",pred)
    protocol.write_json(output/"summary.json",summary);protocol.write_json(output/"report.json",report)
    protocol.write_json(output/"router.json",router);protocol.write_json(output/"crossfit_audit.json",oof)
    _species(output,pred,info["meta"])
    receipt.update(targets_passed=summary["targets_passed"],summary=summary,crossfit_passed=oof["passed"],
        calibration_gate_is_execution_gate=False,test_allowed_after_failed_gates=True)
    artifacts=dict(scores="scores.jsonl",predictions="predictions.jsonl",summary="summary.json",
        report="report.json",router="router.json",crossfit="crossfit_audit.json",species="per_species.csv")
    if fold_scores is not None:
        support._save_torch(output/"oof_scores.pth",fold_scores)
        artifacts["oof_scores"]="oof_scores.pth"
    return _complete(output,receipt,artifacts)


def calibrate_arm(suite,arm_id,device):
    from . import runner
    suite,cfg,info=_checked(suite);arm=next(a for a in cfg["arms"] if a["id"]==arm_id)
    cache,cached=_load_cache(suite,"development",cfg,info)
    traincache,_=_load_cache(suite,"train",cfg,info)
    payload,trained=_load_model(suite,arm,cfg,info)
    groups={k:v["records"] for k,v in cache["groups"].items()} if arm["kind"]=="reference" else training.score(cache,payload,device)
    rows=_rows(groups)
    original=suite/"arms/R00_reference/calibration/crossfit_audit.json"
    snapshot=runner._verify_snapshot(suite,cfg,info)
    if arm["kind"]!="reference" and original.exists():
        runner._verify_stage(suite,"R00_reference","calibration",snapshot)
        folds=protocol.read_json(original)["reference_folds"]
    else:
        folds=calibration.reference_folds(rows,info["meta"],cfg["calibration"],info["config"]["calibration"])
    reused=None
    if arm["root_guard"]:
        source_dir=suite/"arms"/arm["weight_source"]/"calibration"
        if (source_dir/runner.STAGE_MARKER).is_file():
            runner._verify_stage(suite,arm["weight_source"],"calibration",snapshot)
            reused=support._load_torch(source_dir/"oof_scores.pth")
    oof,fold_scores=calibration.source_crossfit(traincache,cache,info["meta"],arm,cfg,folds,device,reused=reused)
    router,diagnostics=calibration.fit(rows,info["meta"],cfg["calibration"],info["router"],arm)
    if arm["kind"]=="reference":
        diagnostics["source_reproduction"]=cache["source_reproduction"]
    receipt=dict(_header(cfg,info,"calibration",arm_id),model_sha256=trained["artifacts"]["model"]["sha256"],
        training_receipt_sha256=protocol.file_hash(suite/"arms"/arm_id/"training/completed.json"),
        cache_sha256=cached["artifacts"]["features"]["sha256"])
    return _export(_claim(suite/"arms"/arm_id/"calibration"),rows,router,arm,info,oof,diagnostics,receipt,fold_scores)


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
