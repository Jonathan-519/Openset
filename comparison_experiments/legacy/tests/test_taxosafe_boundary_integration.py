"""Ten real Boundary arms from a genuinely trained tiny D05.

Only original image/model I/O and process launching are fixture seams. Cached
features, EVM fits, gradients, serialized inference, calibration and reporting
run their production implementations. No server result is simulated here.
"""
import copy
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import traceback
import unittest
from unittest.mock import patch

import torch

from tests import test_taxosafe_discovery_integration as discovery_fixture
from tests import test_taxosafe_refine_pipeline as reference_fixture
from taxosafe_support import pipeline as support
from taxosafe_boundary import backend, importer, protocol, reporting, runner


class BoundaryLifecycle(discovery_fixture.DiscoveryLifecycle):
    legacy_torch = False

    def _boundary_json(self, arm_id, stage, name="completed.json"):
        return protocol.read_json(self.boundary / "arms" / arm_id / stage / name)

    def _boundary_model(self, arm_id):
        return support._load_torch(self.boundary / "arms" / arm_id / "training/model.pth")

    def test_ten_real_arms_fail_gates_still_test_with_frozen_d05_and_no_refitting(self):
        discovery_fixture.DiscoveryLifecycle.test_eleven_real_arms_failed_gates_still_test_without_any_refitting(self)
        discovery = self.suite
        parent_files = reference_fixture.artifact_snapshot(discovery)
        parent = importer.load_d05(discovery)
        parent_digest = backend._semantic(parent.payload)
        source_state = copy.deepcopy(parent.payload["verifier"])
        cfg = copy.deepcopy(protocol.DEFAULTS)
        cfg["training"].update(steps=2, batch_size=16)
        cfg = protocol.validate_config(cfg)
        self.boundary = self.root / "boundary"
        calls, test_reads = [], []
        original_load, original_read = support._load_torch, Path.read_text

        def guarded_load(path):
            if Path(path).resolve() == (discovery / "cache/test/features.pth").resolve():
                self.assertTrue((self.boundary / "dev_selection.json").is_file())
                test_reads.append(str(path))
            return original_load(path)

        def no_old_test_predictions(path, *args, **kwargs):
            try:
                relative = Path(path).resolve().relative_to(discovery.resolve())
            except ValueError:
                relative = None
            if relative is not None and "test" in relative.parts and relative.name in ("predictions.jsonl", "scores.jsonl"):
                raise AssertionError("Boundary read historical TEST predictions")
            return original_read(path, *args, **kwargs)

        def in_process_worker(suite, arm_id, stage, device):
            # Same worker as production; a fresh process cannot inherit the
            # tiny external-data fixture. Lifecycle failure handling stays real.
            calls.append((arm_id, stage))
            logs = suite / "logs" / (arm_id or "cache")
            logs.mkdir(parents=True, exist_ok=True)
            stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
            stdout.write_text("tiny real Boundary integration\n")
            try:
                with ExitStack() as stack:
                    if stage in ("cache_test", "test"):
                        self.assertTrue((suite / "dev_selection.json").is_file())
                        for target in (
                            "taxosafe_boundary.core.BoundaryBank.fit",
                            "taxosafe_boundary.core.build_episodes",
                            "taxosafe_boundary.core.BoundaryVerifier.fit",
                            "taxosafe_boundary.core.make_leaf_guard",
                            "taxosafe_boundary.calibration.fit_router",
                            "taxosafe_boundary.calibration.crossfit_audit",
                            "taxosafe_discovery.geometry.GeometryBank.fit",
                            "taxosafe_discovery.verifier.SharedVerifier.fit",
                            "taxosafe_discovery.verifier.build_episodes",
                            "taxosafe_discovery.calibration.fit_router",
                        ):
                            stack.enter_context(patch(target, side_effect=AssertionError("TEST fitting: " + target)))
                    runner.worker(suite, arm_id, stage, device)
                stderr.write_text("")
                return 0, stdout, stderr
            except Exception:
                stderr.write_text(traceback.format_exc())
                return 2, stdout, stderr

        compatibility = ExitStack()
        if self.legacy_torch:
            from tests.test_taxosafe_boundary_legacy_torch import legacy_torch_apis
            compatibility.enter_context(legacy_torch_apis(isin_mode="missing"))
        with compatibility, patch.object(support, "_load_torch", side_effect=guarded_load), \
                patch.object(Path, "read_text", new=no_old_test_predictions), \
                patch.object(support, "make_loader", side_effect=AssertionError("Boundary must reuse features without image forwarding")), \
                patch.object(runner, "_launch_stage", side_effect=in_process_worker), \
                patch("builtins.print"):
            preflight = runner.preflight(cfg, discovery, self.boundary)
            self.assertFalse(self.boundary.exists())
            self.assertFalse(preflight["test_cache_opened"])
            self.assertEqual(test_reads, [])
            summary = runner.execute_suite(cfg, discovery, self.boundary)
            self.assertEqual(summary["completed_calibration_count"], 10, summary["technical_failures"])
            self.assertEqual(summary["completed_test_count"], 10, summary["technical_failures"])
            self.assertEqual(summary["technical_failures"], {})
            self.assertEqual(len(summary["experiment_matrix"]), 10)
            for phase in ("development", "test"):
                fingerprints = summary["diagnostics"][phase]["fingerprints"]
                self.assertEqual({value["candidate_sha256"] for value in fingerprints.values()},
                                 {fingerprints["G01_d05"]["candidate_sha256"]})
            self.assertTrue(all(not row["dev_targets_passed"] for row in summary["all_arms"]))
            self.assertEqual(summary["recommendation_arm_id"], "G00_reference")
            index = calls.index((None, "cache_test"))
            self.assertEqual(sum(stage == "calibration" for _, stage in calls[:index]), 10)
            self.assertTrue(all(stage == "test" for _, stage in calls[index+1:]))
            self.assertGreater(len(test_reads), 0)
            self.assertEqual(self._boundary_json("G00_reference", "calibration", "router.json"), parent.info["reference"]["router"])
            self.assertEqual(self._boundary_json("G01_d05", "calibration", "router.json"), parent.router)
            reproduction = self._boundary_json("G01_d05", "calibration", "report.json")["calibration_diagnostics"]["source_reproduction"]
            self.assertTrue(all(reproduction[k] for k in ("exact_scores", "exact_candidates", "exact_terminal_outputs")))
            for arm_id in ("G04_bce8", "G05_rank8", "G06_bce9", "G07_rank9"):
                payload = self._boundary_model(arm_id)
                state = payload["boundary_verifier"]
                self.assertEqual(backend._semantic(payload["parent_payload"]), parent_digest)
                self.assertEqual(backend._semantic(state["source_state"]), backend._semantic(source_state))
                self.assertEqual(self._boundary_json(arm_id, "training")["optimizer_steps"], 4)
                expected_dimension = 8 if arm_id.endswith("8") else 9
                self.assertEqual(state["dimensions"], dict(leaf=expected_dimension, parent=expected_dimension))
                for level in ("leaf", "parent"):
                    self.assertNotEqual(backend._semantic(state["heads"][level]), backend._semantic(source_state["heads"][level]))
                    for key in ("mean", "scale"):
                        self.assertTrue(torch.equal(state["normalization"][level][key][:8], source_state["normalization"][level][key]))
            for original, reused in (("G01_d05", "G02_d05_kp"), ("G07_rank9", "G08_rank9_standard")):
                self.assertEqual(self._boundary_json(original, "training")["model"], self._boundary_json(reused, "training")["model"])
                self.assertEqual(self._boundary_json(reused, "training")["optimizer_steps"], 0)
            g07, guard = self._boundary_model("G07_rank9"), self._boundary_model("G09_rank9_leaf_guard")
            self.assertEqual(backend._semantic(guard["boundary_bank"]), backend._semantic(g07["boundary_bank"]))
            self.assertEqual(backend._semantic(guard["boundary_verifier"]["heads"]["leaf"]), backend._semantic(g07["boundary_verifier"]["heads"]["leaf"]))
            self.assertEqual(backend._semantic(guard["boundary_verifier"]["heads"]["parent"]), backend._semantic(source_state["heads"]["parent"]))
            selection_digest = protocol.file_hash(self.boundary / "dev_selection.json")
            freeze = datetime.fromisoformat(summary["dev_selection"]["created_at_utc"])
            for arm in cfg["arms"]:
                arm_id = arm["id"]
                receipt = self._boundary_json(arm_id, "test")
                self.assertFalse(receipt["calibration_gate_is_execution_gate"])
                self.assertTrue(receipt["test_allowed_after_failed_gates"])
                self.assertEqual(receipt["dev_selection_sha256"], selection_digest)
                records = reference_fixture.read_records(self.boundary / "arms" / arm_id / "test/predictions.jsonl")
                self.assertEqual(len(records), 11)
                self.assertEqual(sum(row["evaluation_weight"] for row in records), 10)
                for stage in ("training", "calibration", "test"):
                    marker = self._boundary_json(arm_id, stage, "stage_binding.json")
                    self.assertGreaterEqual(marker["elapsed_seconds"], 0)
                    if stage == "test":
                        self.assertGreaterEqual(datetime.fromisoformat(marker["started_at_utc"]), freeze)
            for phase in ("calibration", "test"):
                self.assertTrue(all(self._boundary_json("G09_rank9_leaf_guard", phase, "report.json")["leaf_guard_preservation"].values()))
            original_calls = list(calls)
            runner.execute_suite(cfg, discovery, self.boundary, resume=True)
            self.assertEqual(calls, original_calls)
            self.assertEqual(protocol.file_hash(self.boundary / "dev_selection.json"), selection_digest)
        self.assertEqual(backend._semantic(parent.payload), parent_digest)
        self.assertEqual(reference_fixture.artifact_snapshot(discovery), parent_files)
        self._assert_source_unchanged()


for _name in dir(discovery_fixture.DiscoveryLifecycle):
    if _name.startswith("test_") and _name not in BoundaryLifecycle.__dict__:
        setattr(BoundaryLifecycle, _name, None)


class BoundaryLegacyTorchLifecycle(BoundaryLifecycle):
    """Same genuine ten-arm pipeline when stable sorting and torch.isin lack API support."""
    legacy_torch = True


if __name__ == "__main__":
    unittest.main()
