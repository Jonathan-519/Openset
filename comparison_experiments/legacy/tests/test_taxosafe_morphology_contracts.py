"""Deployment contracts and explicit failure isolation for morphology."""
import ast
import copy
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_morphology import protocol, diagnostics, reporting, runner
from tools.pack_taxosafe_morphology_review import pack
from tests.test_taxosafe_morphology_calibration import as_morphology
from tests.test_taxosafe_discovery_calibration import fixture


class MorphologyContracts(unittest.TestCase):
    def test_predeclared_matrix_and_invalid_options(self):
        cfg=protocol.effective_config(protocol.PROJECT_ROOT/protocol.DEFAULT_CONFIG)
        self.assertEqual(cfg,protocol.DEFAULTS)
        self.assertEqual([a["mode"] for a in cfg["arms"]],["C00","C01","C02","R01","L01"])
        for update in (lambda c:c["arms"].pop(),lambda c:c.update(skip_failed_gates=True),
                       lambda c:c["calibration"].update(root_known_target=.9),
                       lambda c:c["training"].update(steps=True),
                       lambda c:c["matching"].update(epsilon=float("nan")),
                       lambda c:c["features"].update(view="test_selected_crop")):
            changed=copy.deepcopy(cfg);update(changed)
            with self.assertRaises(ValueError): protocol.validate_config(changed)

    def test_python38_syntax_and_no_new_torch_only_calls(self):
        for path in (protocol.PROJECT_ROOT/"taxosafe_morphology").glob("*.py"):
            source=path.read_text();ast.parse(source,feature_version=(3,8))
            self.assertNotIn("stable=True",source)
            self.assertNotIn("torch.isin",source)
            self.assertNotIn("torch.compile",source)

    def test_new_code_is_signed_without_changing_historical_contract(self):
        from taxosafe_domain import protocol as previous
        old=set(previous.code_files());new=set(protocol.code_files())
        self.assertTrue(old < new)
        self.assertFalse(any("taxosafe_morphology" in str(p) for p in old))
        one=protocol.signature(protocol.DEFAULTS,{"parent":"a"})
        other=copy.deepcopy(protocol.DEFAULTS);other["training"]["steps"]+=1
        self.assertNotEqual(one,protocol.signature(other,{"parent":"a"}))
        self.assertNotEqual(one,protocol.signature(protocol.DEFAULTS,{"parent":"b"}))

    def test_threshold_bound_is_descriptive_and_not_searched_on_test(self):
        groups=as_morphology(fixture());rows=sum(groups,[])
        result=diagnostics.score_diagnostics(rows)
        self.assertFalse(result["changes_executable_router"])
        self.assertIsNotNone(result["necessary_DEV_bound"])
        for row in rows:row["split"]=row["split"].replace("val_","test_")
        self.assertIsNone(diagnostics.score_diagnostics(rows)["necessary_DEV_bound"])

    def test_root_and_leaf_isolation_checks_detect_changed_pass_set(self):
        values={"root_score_sha256":"a","root_state_sha256":"b","root_pass_sha256":"c","leaf_score_sha256":"d"}
        fp={name:dict(values) for name in ("C02_d05_staged","L01_spatial_leaf","R01_spatial_parent")}
        self.assertEqual(reporting._root_invariants(fp)["validation"],"verified")
        fp["L01_spatial_leaf"]["root_pass_sha256"]="tampered"
        with self.assertRaises(ValueError):reporting._root_invariants(fp)

    def test_technical_failed_arm_is_preserved_and_other_arms_test(self):
        cfg=copy.deepcopy(protocol.DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";reference=root/"reference";suite=root/"suite"
            source.mkdir();reference.mkdir()
            info=dict(directory=source,reference=dict(directory=reference),binding=dict(directory=str(source)))
            calls=[]
            def stage(suite,cfg,source,arm,phase,device,resume=False):
                calls.append((arm,phase))
                if arm=="R01_spatial_parent" and phase=="training":
                    runner._failure(suite,arm,phase,"intentional technical failure")
                    return False
                return True
            def freeze(suite):
                protocol.write_json(suite/"dev_selection.json",dict(frozen=True))
            with patch.object(runner,"_source",return_value=info),patch.object(runner,"_execute",side_effect=stage), \
                    patch.object(reporting,"freeze_dev_selection",side_effect=freeze), \
                    patch.object(reporting,"summarize_suite",return_value={}),patch("builtins.print"):
                runner.execute_suite(cfg,source,suite,run_preflight=False)
            self.assertNotIn(("R01_spatial_parent","test"),calls)
            self.assertIn(("L01_spatial_leaf","test"),calls)
            finished=protocol.read_json(suite/"suite_completed.json")
            self.assertEqual(finished["technical_failure_arms"],["R01_spatial_parent"])
            self.assertEqual(len(finished["completed_test_arms"]),4)
            self.assertTrue(finished["all_valid_calibrations_test_attempted_regardless_of_gate"])

    def test_cli_and_review_preserve_raw_audit_but_exclude_models(self):
        result=subprocess.run([sys.executable,"-m","taxosafe_morphology","--help"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn("--preflight",result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/"suite";cache=suite/"cache/train";cache.mkdir(parents=True)
            protocol.write_json(suite/"snapshot.json",dict(schema_version=protocol.SCHEMA_VERSION))
            protocol.write_json(cache/"raw_image_audit.json",dict(valid=False,problems=["missing image"]))
            (cache/"features.pth").write_bytes(b"must not be exported")
            with patch("sys.stdout",new=io.StringIO()):archive=pack(suite)
            with tarfile.open(archive) as handle:
                names=handle.getnames()
            self.assertIn("suite/cache/train/raw_image_audit.json",names)
            self.assertFalse(any(name.endswith(".pth") for name in names))


if __name__=="__main__":
    unittest.main()
