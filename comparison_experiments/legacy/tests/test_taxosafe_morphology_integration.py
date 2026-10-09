"""Five full lifecycles over a genuinely trained tiny reference and D05.

Only external image I/O and process launch are synthetic. Production spatial
extraction, matching gradients, serialized inference, source auditing, OOF and
quality-failed TEST all execute. This does not measure zooplankton accuracy.
"""
import copy
from contextlib import ExitStack
from pathlib import Path
import traceback
import unittest
import warnings
from unittest.mock import patch

import torch

from tests import test_taxosafe_discovery_integration as fixture
from tests import test_taxosafe_refine_pipeline as reference_fixture
from tests.test_taxosafe_boundary_legacy_torch import legacy_torch_apis
from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_morphology import backend, features, importer, protocol, runner, training


class TinySpatialBackbone(fixture.TinyCoreBackbone):
    def encode_image_with_spatial(self, images, normalize=True):
        global_features, spatial = super().encode_image_with_spatial(images, normalize)
        return global_features, spatial.repeat(1,2,1)


class MorphologyLifecycle(fixture.DiscoveryLifecycle):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.object(support,"make_backbone",side_effect=lambda *args:TinySpatialBackbone()))
        self.stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore",message="A single label was found in",category=UserWarning)
        warnings.filterwarnings("ignore",message="y_pred contains classes not in y_true",category=UserWarning)

    def test_five_real_arms_frozen_root_and_failed_quality_still_test(self):
        fixture.DiscoveryLifecycle.test_eleven_real_arms_failed_gates_still_test_without_any_refitting(self)
        discovery = self.suite
        old_files=reference_fixture.artifact_snapshot(discovery)
        original=importer.load_d05(discovery)
        cfg=copy.deepcopy(protocol.DEFAULTS)
        cfg["support"]["folds"]=2
        cfg["training"].update(steps=4,batch_size=2,adapter_dim=4,hidden=8,log_every=4)
        cfg["matching"]["iterations"]=8
        morphology=self.root/"morphology"
        self.morphology=morphology
        calls=[]

        def synthetic_image_audit(cache, info, decode=True):
            if any(k.startswith("test_") for k in cache["groups"]):
                self.assertTrue((morphology/"dev_selection.json").is_file())
            groups={}
            for name,group in cache["groups"].items():
                rows={row["image_sha256"]:row for row in base.unique_records(group["records"])}
                groups[name]=[rows[h] for h in group["image_sha256"]]
            return groups,dict(valid=True,image_count=sum(len(v) for v in groups.values()),problems=[],
                               inventory={},test_images_opened=any(k.startswith("test_") for k in groups))

        def launch(suite, arm, stage, device):
            calls.append((arm,stage))
            logs=suite/"logs"/(arm or "cache");logs.mkdir(parents=True,exist_ok=True)
            stdout,stderr=logs/(stage+".stdout.log"),logs/(stage+".stderr.log")
            stdout.write_text("Tiny real morphology lifecycle\n")
            try:
                with ExitStack() as stack:
                    if stage in ("cache_test","test"):
                        self.assertTrue((suite/"dev_selection.json").is_file())
                        for name in ("taxosafe_morphology.training.fit_spatial", "taxosafe_morphology.episodes.build_episodes",
                                     "taxosafe_discovery.geometry.GeometryBank.fit", "taxosafe_discovery.verifier.SharedVerifier.fit",
                                     "taxosafe_morphology.calibration.fit_router", "taxosafe_morphology.calibration.crossfit_audit"):
                            stack.enter_context(patch(name,side_effect=AssertionError("TEST fitting: "+name)))
                    runner.worker(suite,arm,stage,device)
                stderr.write_text("");return 0,stdout,stderr
            except Exception:
                stderr.write_text(traceback.format_exc());return 2,stdout,stderr

        with legacy_torch_apis(isin_mode="missing"), patch.object(features,"audited_rows",side_effect=synthetic_image_audit), \
                patch.object(support,"make_loader",side_effect=fixture.image_loader), \
                patch.object(runner,"_launch_stage",side_effect=launch):
            checked=runner.preflight(cfg,discovery,morphology)
            self.assertFalse(morphology.exists())
            self.assertFalse(checked["test_cache_opened"])
            result=runner.execute_suite(cfg,discovery,morphology)
            self.assertEqual(result["technical_failures"],{})
            self.assertEqual(result["completed_test_count"],5)
            self.assertEqual(result["recommendation_arm_id"],"C00_reference")
            self.assertTrue(all(not row["dev_targets_passed"] for row in result["all_arms"]))
            self.assertEqual(sum(s=="calibration" for _,s in calls[:calls.index((None,"cache_test"))]),5)
            for arm in cfg["arms"]:
                current=morphology/"arms"/arm["id"]
                report=protocol.read_json(current/"training/training_report.json")
                self.assertEqual(report["optimizer_steps"],4 if arm["kind"]=="fit" else 0)
                if arm["kind"]=="fit":
                    self.assertGreater(report["parameter_delta_l2"],0.)
                    self.assertEqual(report["initial_control_max_abs_delta"],0.)
                test=protocol.read_json(current/"test/completed.json")
                self.assertTrue(test["test_allowed_after_failed_gates"])
                self.assertFalse(test["calibration_gate_is_execution_gate"])
                rows=reference_fixture.read_records(current/"test/predictions.jsonl")
                self.assertEqual(len(rows),11)
                self.assertEqual(sum(r["evaluation_weight"] for r in rows),10)
            for stage in ("calibration","test"):
                controls=[protocol.read_json(morphology/"arms"/a/stage/"router.json") for a in ("C02_d05_staged","L01_spatial_leaf")]
                self.assertEqual(controls[0]["root_state"],controls[1]["root_state"])
                self.assertEqual(controls[0]["root_threshold"],controls[1]["root_threshold"])
            c02=protocol.read_json(morphology/"arms/C02_d05_staged/calibration/crossfit_audit.json")
            l01=protocol.read_json(morphology/"arms/L01_spatial_leaf/calibration/crossfit_audit.json")
            for a,b in zip(c02["folds"],l01["folds"]):
                self.assertEqual(a.get("root_state_sha256"),b.get("root_state_sha256"))
            _,_,info=backend._checked(morphology)
            cached,_=backend._load_cache(morphology,"test",cfg,info)
            changed=copy.deepcopy(cached)
            for group in changed["groups"].values():
                for row in group["records"]:
                    row.update(status="extra",true_leaf=None,true_parent=None)
            for arm in cfg["arms"][-2:]:
                state,_=backend._load_model(morphology,arm,cfg,info)
                before=backend.score_groups(cached,arm,state,info["meta"])
                after=backend.score_groups(changed,arm,state,info["meta"])
                for split in before:
                    self.assertEqual([r["morphology"] for r in before[split]],[r["morphology"] for r in after[split]])
            executed=list(calls)
            runner.execute_suite(cfg,discovery,morphology,resume=True)
            self.assertEqual(calls,executed)
            with patch.object(protocol,"code_signature",return_value="changed"):
                with self.assertRaises(ValueError):
                    runner.execute_suite(cfg,discovery,morphology,resume=True)
            p=morphology/"arms/L01_spatial_leaf/calibration/router.json"
            value=p.read_bytes();p.write_bytes(value+b" ")
            with self.assertRaises(ValueError):
                runner.execute_suite(cfg,discovery,morphology,resume=True)
            p.write_bytes(value)
        self.assertEqual(reference_fixture.artifact_snapshot(discovery),old_files)
        for old,new in (("D00_reference","C00_reference"),("D05_episode_bce","C01_d05")):
            for phase in ("calibration","test"):
                a=reference_fixture.read_records(discovery/"arms"/old/phase/"predictions.jsonl")
                b=reference_fixture.read_records(morphology/"arms"/new/phase/"predictions.jsonl")
                self.assertEqual([{k:r[k] for k in ("image_sha256","prediction_type","parent","leaf","output_node")} for r in a],
                                 [{k:r[k] for k in ("image_sha256","prediction_type","parent","leaf","output_node")} for r in b])
        self._assert_source_unchanged()


for name in dir(fixture.DiscoveryLifecycle):
    if name.startswith("test_") and name not in MorphologyLifecycle.__dict__:
        setattr(MorphologyLifecycle,name,None)


if __name__=="__main__":
    unittest.main()
