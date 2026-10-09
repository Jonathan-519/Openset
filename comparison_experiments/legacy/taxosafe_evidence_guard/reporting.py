"""Recomputed metrics, immutable DEV recommendation, and complete arm comparison."""
import csv
import json
from pathlib import Path
from taxosafe_support import calibration as base
from . import protocol,calibration


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _stage(root,arm,stage,snapshot):
    from . import runner
    receipt=runner._verify_stage(root,arm,stage,snapshot)
    directory=Path(root)/"arms"/arm/stage
    pred=_read_records(directory/"predictions.jsonl")
    summary=base.evaluate_records(pred,receipt["meta"])
    if summary!=receipt["summary"] or summary!=protocol.read_json(directory/"summary.json"):
        raise ValueError("Saved metrics do not reproduce actual predictions: "+arm+"/"+stage)
    if sum(r.get("evaluation_weight",0) for r in pred)!=summary["unique_image_count"]:
        raise ValueError("Duplicate weighting changed")
    report=protocol.read_json(directory/"report.json")
    if any(report.get(k)!=v for k,v in summary.items()):
        raise ValueError("Report metrics differ from predictions")
    if stage=="test":
        frozen=Path(root)/"dev_selection.json"
        if receipt["dev_selection_sha256"]!=protocol.file_hash(frozen):
            raise ValueError("TEST source DEV freeze changed")
        original=Path(root)/"arms"/arm/"calibration"
        if (receipt["router_sha256"]!=protocol.file_hash(original/"router.json") or
                protocol.read_json(directory/"router.json")!=protocol.read_json(original/"router.json")):
            raise ValueError("TEST router differs from frozen DEV router")
    return dict(receipt=receipt,summary=summary,report=report,predictions=pred)


def _development(root):
    from .backend import _checked
    root,cfg,info=_checked(root)
    snapshot=protocol.read_json(root/"snapshot.json")
    loaded={};inventory={};failures={}
    for arm in cfg["arms"]:
        name=arm["id"];directory=root/"arms"/name/"calibration"
        if (directory/"stage_binding.json").is_file():
            value=_stage(root,name,"calibration",snapshot);loaded[name]=value
            inventory[name]=dict(completed_sha256=protocol.file_hash(directory/"completed.json"),
                stage_binding_sha256=protocol.file_hash(directory/"stage_binding.json"))
        else:
            failure=root/"arms"/name/"failure.json"
            error=protocol.read_json(failure) if failure.is_file() else dict(error="calibration unavailable")
            inventory[name]=dict(unavailable=True,error=error)
            failures[name]=error
    baseline=loaded.get("R00_reference")
    qualified=[];entries=[]
    for name,value in loaded.items():
        summary=value["summary"]
        preserve=bool(baseline and summary["counts"]["known_correct"]>=baseline["summary"]["counts"]["known_correct"])
        macro_preserve=bool(baseline and calibration.macro(summary,"per_known_leaf")+1e-12>=calibration.macro(baseline["summary"],"per_known_leaf"))
        extra_preserve=bool(baseline and summary["counts"]["extra_correct"]>=baseline["summary"]["counts"]["extra_correct"])
        oof_complete=bool(value["report"].get("crossfit_complete",False))
        oof_passed=bool(value["report"].get("crossfit_passed",False))
        passed=bool(summary["targets_passed"] and preserve and macro_preserve and extra_preserve and oof_complete and oof_passed)
        entry=dict(arm_id=name,metrics=summary["metrics"],counts=summary["counts"],targets_passed=summary["targets_passed"],
                   crossfit_passed=oof_passed,crossfit_complete=oof_complete,known_count_preserved=preserve,
                   known_macro_preserved=macro_preserve,extra_count_preserved=extra_preserve,qualified=passed)
        entries.append(entry)
        if name!="R00_reference" and passed:qualified.append(entry)
    qualified.sort(key=lambda a:(-min(a["metrics"].values()),-a["counts"]["known_correct"],a["arm_id"]))
    recommendation=qualified[0]["arm_id"] if qualified else "R00_reference" if baseline else None
    return dict(schema_version="evidence_guard_DEV_selection_v1",suite_signature=snapshot["signature"],
        snapshot_sha256=protocol.file_hash(root/"snapshot.json"),inventory=inventory,entries=entries,
        recommendation_arm_id=recommendation,qualified_candidate_arm_id=qualified[0]["arm_id"] if qualified else None,
        recommendation_status="qualified_DEV_candidate" if qualified else "retain_C00_no_qualified_candidate" if baseline else "no_valid_reference",
        technical_failures=failures,selection_uses_test=False,test_predictions_read=False,
        ranking_rule="all four DEV gates AND complete adapter source-heldout OOF passed AND C00 known count/macro and extra ROOT count preserved; then max worst metric, known count, arm ID",
        confirmatory_validation=False,benchmark_reused=True,model_selection_used_TEST=False)


def freeze_dev_selection(root):
    root=Path(root);value=_development(root);path=root/"dev_selection.json"
    if path.exists():
        if protocol.read_json(path)!=value:raise ValueError("Frozen DEV artifacts or selection changed")
    else:protocol.write_json(path,value)
    return value


def _csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def _training_counts(report):
    """Distinguish eligible fit images, actually sampled images, and draws.

    Empty/missing training metadata is not fabricated as a successfully trained
    model. An inherited zero-step reference used no images for *new* gradients;
    its original C00 training history is a separate provenance record.
    """
    counts=report.get("fit_status_counts",{})
    eligible=report.get("fit_image_sha256")
    used=report.get("used_image_sha256")
    by_split=report.get("used_train_image_sha256",{})
    draws=report.get("sampled_status_counts",{})
    zero_step=report.get("optimizer_steps")==0
    result=dict(train_eligible_unique_images=len(set(eligible)) if eligible is not None else None,
                train_used_unique_images=len(set(used)) if used is not None else 0 if zero_step else None)
    for status,label,split in (("known","known","train"),("intra","near","train_intra"),("extra","extra","oe_train")):
        result["train_eligible_"+label+"_images"]=int(counts.get(status,0)) if "fit_status_counts" in report else None
        values=by_split.get(split)
        result["train_used_"+label+"_images"]=len(set(values)) if values is not None else 0 if zero_step else None
        result["train_sampled_"+label+"_draws"]=int(draws.get(status,0)) if "sampled_status_counts" in report else 0 if zero_step else None
    return result


def summarize_suite(root,phase="complete"):
    from .backend import _checked
    root,cfg,info=_checked(root);selection=freeze_dev_selection(root)
    snapshot=protocol.read_json(root/"snapshot.json")
    table=[];species=[];failures={};details={};completed=dict(calibration=0,test=0)
    dev_entries={entry["arm_id"]:entry for entry in selection["entries"]}
    for arm in cfg["arms"]:
        name=arm["id"]
        failure=root/"arms"/name/"failure.json"
        if failure.is_file():failures[name]=protocol.read_json(failure)
        trained=root/"arms"/name/"training/training_report.json"
        train=protocol.read_json(trained) if trained.is_file() else {}
        train_counts=_training_counts(train)
        dev_entry=dev_entries.get(name,{})
        for stage in ("calibration","test"):
            directory=root/"arms"/name/stage
            row=dict(arm_id=name,phase=stage,execution="unavailable",optimizer_steps=train.get("optimizer_steps"),
                     use_oe=bool(arm.get("use_oe",False)),
                     source_training_expanded=train.get("source_training_expanded"),
                     targets_passed=None,crossfit_passed=None,crossfit_complete=None,
                     DEV_qualified=dev_entry.get("qualified"),
                     DEV_known_count_preserved=dev_entry.get("known_count_preserved"),
                     DEV_known_macro_preserved=dev_entry.get("known_macro_preserved"),
                     DEV_extra_ROOT_count_preserved=dev_entry.get("extra_count_preserved"),
                     recommended_by_DEV=name==selection["recommendation_arm_id"],**train_counts,
                     **{k:None for k in base.TARGETS})
            if (directory/"stage_binding.json").is_file():
                value=_stage(root,name,stage,snapshot);completed[stage]+=1
                summary=value["summary"];report=value["report"]
                row.update(execution="completed",targets_passed=summary["targets_passed"],crossfit_passed=report.get("crossfit_passed",False),
                           crossfit_complete=report.get("crossfit_complete",False),
                           **summary["metrics"],**summary["counts"])
                row.update(known_macro=calibration.macro(summary,"per_known_leaf"),near_macro=calibration.macro(summary,"per_intra_species"),
                           extra_macro=calibration.macro(summary,"per_extra_source"))
                details[name+"/"+stage]=dict(summary=summary,paired_to_C00=report["paired_to_C00"],
                    use_oe=bool(arm.get("use_oe",False)),source_training_expanded=train.get("source_training_expanded"),
                    training_counts=train_counts)
                for status,key in (("known","per_known_leaf"),("intra","per_intra_species"),("extra","per_extra_source")):
                    for species_name,item in summary[key].items():
                        species.append(dict(arm_id=name,phase=stage,status=status,species=species_name,
                            sample_count=item["sample_count"],correct_count=item["correct_count"],correct_terminal_rate=item["correct_rate"],
                            leaf_acceptance_rate=item["leaf_acceptance_rate"],parent_fallback_rate=item["parent_fallback_rate"],root_rejection_rate=item["root_rejection_rate"]))
            table.append(row)
    result=dict(schema_version="evidence_guard_comparison_v1",phase=phase,declared_arm_count=len(cfg["arms"]),
        completed_calibration_count=completed["calibration"],completed_test_count=completed["test"],
        workflow_completed=phase=="complete",all_arms=table,details=details,technical_failures=failures,
        recommendation_arm_id=selection["recommendation_arm_id"],recommendation_status=selection["recommendation_status"],
        dev_selection_sha256=protocol.file_hash(root/"dev_selection.json"),production_selection_uses_test=False,
        confirmatory_validation=False,validation_scope="exploratory on previously reviewed DEV/TEST; source-heldout adapter OOF conditional on fixed C00",
        training_regime_note="use_oe arms add real near/extra TRAIN images; these are not known-only C00 training comparisons",
        independent_full_pipeline_validation=False,
        evaluation_rule="known -> correct leaf; near -> correct parent only; extra -> root only")
    protocol.write_json(root/"summary.json",result);_csv(root/"comparison_all.csv",table);_csv(root/"per_species_comparison.csv",species)
    lines=["# C00 evidence-preserving ablations","", "DEV recommendation: "+str(selection["recommendation_arm_id"]),
           "", "Near success requires the correct parent. Extra success requires root. All quality-failed calibrations still run TEST.",
           "DEV qualification requires all four targets, C00 known count/macro and extra ROOT count preservation, and complete passed OOF.",
           "Source-heldout adapter OOF is conditional on the frozen C00 encoder/verifiers/support, not independent full-pipeline validation.",
           "TEST never changes the recommendation. use_oe arms expand real TRAIN data and must be interpreted as a different supervision regime.",
           "", "| Arm | Phase | Execution | use_oe | Used TRAIN images | Known | Near parent | Extra root | Leaf precision | Gates |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in table:
        rates=["NA" if row[k] is None else "{:.2f}%".format(100*row[k]) for k in base.TARGETS]
        lines.append("| "+" | ".join([row["arm_id"],row["phase"],row["execution"],str(row["use_oe"]),str(row["train_used_unique_images"]),*rates,str(row["targets_passed"])])+" |")
    (root/"comparison_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
