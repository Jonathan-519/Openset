"""All eight real CPU lifecycles on a tiny trained C00 fixture, not user accuracy.

Only external image I/O, the tiny backbone and process launch are substituted.
Production fitting, masking, serialization, calibration, OOF, TEST and receipts
execute; the original baseline's artifact bytes must remain identical.
"""
import copy
from contextlib import ExitStack,redirect_stdout
import io
import json
from pathlib import Path
import traceback
import unittest
from unittest.mock import patch
from tests import test_taxosafe_refine_pipeline as fixture
from tests.test_taxosafe_boundary_legacy_torch import legacy_torch_apis
from taxosafe_support import pipeline as support
from taxosafe_reference_joint import backend,calibration,features,protocol,reporting,runner,training
from tools.pack_taxosafe_reference_joint_review import pack


class ReferenceJointLifecycle(fixture.FrozenPipelineContracts):
    def test_eight_arms_true_updates_failed_gates_still_TEST_and_no_refit(self):
        # Deliberately inseparable known/unknown vectors: failing quality is expected.
        for prefix in ("val","test"):
            known=self.groups[prefix+"_known"]
            for status in ("intra","extra"):
                for i,row in enumerate(self.groups[prefix+"_"+status]):
                    row["vector"]=list(known[i%len(known)]["vector"])
        self.make_source(with_test=True)
        source_bytes=fixture.artifact_snapshot(self.source)
        cfg=copy.deepcopy(protocol.DEFAULTS)
        cfg["support"].update(modes=2,folds=2)
        cfg["training"].update(steps=5,batch_size=8,hidden=8,adapter_dim=4,log_every=5)
        suite=self.root/"joint_suite";calls=[]

        def launch(directory,arm,stage,device):
            calls.append((arm,stage));logs=directory/"logs"/(arm or "cache");logs.mkdir(parents=True,exist_ok=True)
            out,err=logs/(stage+".stdout.log"),logs/(stage+".stderr.log")
            try:
                with out.open("w") as handle,redirect_stdout(handle),ExitStack() as stack:
                    if stage in ("cache_test","test"):
                        self.assertTrue((suite/"dev_selection.json").is_file())
                        for name in ("taxosafe_reference_joint.training.fit","taxosafe_reference_joint.memory.build_memory",
                                     "taxosafe_reference_joint.calibration.fit","taxosafe_reference_joint.calibration.reference_folds",
                                     "taxosafe_reference_joint.training._tail_distribution"):
                            stack.enter_context(patch(name,side_effect=AssertionError("TEST fit forbidden: "+name)))
                    runner.worker(directory,arm,stage,device)
                err.write_text("");return 0,out,err
            except Exception:
                err.write_text(traceback.format_exc());return 2,out,err

        with patch.object(features,"audit_images",return_value=dict(valid=True,problems=[],image_count=0)), \
                patch.object(runner,"_launch_stage",side_effect=launch),legacy_torch_apis(isin_mode="missing"):
            checked=runner.preflight(cfg,self.source,suite)
            self.assertFalse(checked["test_cache_opened"]);self.assertFalse(suite.exists())
            result=runner.execute_suite(cfg,self.source,suite)
            if result["technical_failures"]:
                self.fail(json.dumps(result["technical_failures"],indent=2)+"\nCACHE FAILURES:\n"+
                          "\n".join(p.read_text() for p in (suite/"cache").glob("*/failure.json")))
            self.assertEqual(result["completed_calibration_count"],8);self.assertEqual(result["completed_test_count"],8)
            self.assertTrue(all(not a["targets_passed"] for a in result["all_arms"]))
            self.assertEqual(result["recommendation_arm_id"],"A00_reference")
            first_test=calls.index((None,"cache_test"))
            self.assertEqual(sum(stage=="calibration" for _,stage in calls[:first_test]),8)
            for arm in cfg["arms"]:
                directory=suite/"arms"/arm["id"]
                report=protocol.read_json(directory/"training/training_report.json")
                self.assertEqual(report["optimizer_steps"],0 if arm["kind"]=="reference" else 5)
                if arm["kind"]!="reference":self.assertGreater(report["parameter_delta_l2"],0.)
                if arm["adapter"]:self.assertGreater(report["fine_adapter_delta_l2"],0.)
                self.assertFalse(report["frozen_encoder_updated"])
                test=protocol.read_json(directory/"test/completed.json")
                self.assertTrue(test["test_allowed_after_failed_gates"])
                self.assertEqual(test["summary"]["unique_image_count"],10)
                self.assertEqual(test["summary"]["input_record_count"],11)
            for phase,original in (("calibration","calibration"),("test","test")):
                filename="development_predictions.jsonl" if phase=="calibration" else "predictions.jsonl"
                expected=fixture.read_records(self.source/original/filename)
                actual=fixture.read_records(suite/"arms/A00_reference"/phase/"predictions.jsonl")
                fields=("image_sha256","prediction_type","parent","leaf","output_node")
                self.assertEqual([{k:r[k] for k in fields} for r in expected],[{k:r[k] for k in fields} for r in actual])
            cached,_=backend._load_cache(suite,"test",cfg,runner._source(self.source))
            payload,_=backend._load_model(suite,cfg["arms"][-1],cfg,runner._source(self.source))
            changed=copy.deepcopy(cached)
            for group in changed["groups"].values():
                for row in group["records"]:row.update(status="extra",true_leaf=None,true_parent=None,source="never_seen")
            a,b=training.score(cached,payload),training.score(changed,payload)
            for split in a:self.assertEqual([r["joint_logits"] for r in a[split]],[r["joint_logits"] for r in b[split]])
            calls_before=list(calls);runner.execute_suite(cfg,self.source,suite,resume=True)
            self.assertEqual(calls,calls_before)
            frozen=suite/"dev_selection.json";frozen_bytes=frozen.read_bytes();frozen.unlink()
            with self.assertRaises(ValueError):runner._resume_audit(suite,cfg,runner._source(self.source))
            frozen.write_bytes(frozen_bytes)
            p=suite/"arms/A02_distilled_adapter/calibration/router.json";original=p.read_bytes()
            p.write_bytes(original+b" ")
            with self.assertRaises(ValueError):reporting.freeze_dev_selection(suite)
            p.write_bytes(original)
            archive=pack(suite,self.root/"review.tar.gz")
            import tarfile
            with tarfile.open(archive) as handle:
                self.assertFalse(any(m.name.endswith('.pth') for m in handle.getmembers()))
            print("C00 JOINT CPU FIXTURE: 8 DEV + 8 TEST completed; 7 real learned models; no TEST fitting; source unchanged",flush=True)
        self.assertEqual(fixture.artifact_snapshot(self.source),source_bytes)

    def test_one_technical_failure_does_not_prevent_other_seven_tests(self):
        self.make_source()
        cfg=copy.deepcopy(protocol.DEFAULTS)
        cfg["support"].update(modes=2,folds=2)
        cfg["training"].update(steps=2,batch_size=8,hidden=8,adapter_dim=4,log_every=2)
        suite=self.root/"one_failed_arm";calls=[]
        def launch(directory,arm,stage,device):
            calls.append((arm,stage));logs=directory/"logs"/(arm or "cache");logs.mkdir(parents=True,exist_ok=True)
            out,err=logs/(stage+".stdout.log"),logs/(stage+".stderr.log")
            try:
                with out.open("w") as handle,redirect_stdout(handle):
                    if arm=="A03_no_distillation" and stage=="training":
                        raise RuntimeError("injected optimizer failure")
                    runner.worker(directory,arm,stage,device)
                err.write_text("");return 0,out,err
            except Exception:
                err.write_text(traceback.format_exc());return 2,out,err
        with patch.object(features,"audit_images",return_value=dict(valid=True,problems=[],image_count=0)), \
                patch.object(runner,"_launch_stage",side_effect=launch):
            result=runner.execute_suite(cfg,self.source,suite)
        self.assertEqual(set(result["technical_failures"]),{"A03_no_distillation"})
        self.assertEqual(result["completed_test_count"],7)
        self.assertNotIn(("A03_no_distillation","test"),calls)
        failed=[x for x in result["all_arms"] if x["arm_id"]=="A03_no_distillation"]
        self.assertTrue(all(x["execution"]=="unavailable" and x["known_end_to_end_leaf_accuracy"] is None for x in failed))
        # Do not silently resume an incomplete failed arm or replace it by C00.
        with self.assertRaises(ValueError):runner._resume_audit(suite,cfg,runner._source(self.source))


for name in dir(fixture.FrozenPipelineContracts):
    if name.startswith("test_") and name not in ReferenceJointLifecycle.__dict__:
        setattr(ReferenceJointLifecycle,name,None)


if __name__=="__main__":unittest.main()
