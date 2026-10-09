"""Twelve real statistical Domain arms from a genuinely trained tiny D05.

Only external image/model I/O and subprocess launching are synthetic seams.
The new lifecycle runs with the observed unavailable old-Torch APIs removed.
Missing eigh is an additional fallback stress check, not a claim about the
server's now-recorded Torch 1.12.1 installation.
This checks execution and contracts, not accuracy on the server benchmark.
"""
import copy
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import traceback
import unittest
import warnings
from unittest.mock import patch

import torch

from tests import test_taxosafe_discovery_integration as discovery_fixture
from tests import test_taxosafe_refine_pipeline as reference_fixture
from tests.test_taxosafe_boundary_legacy_torch import legacy_torch_apis
from taxosafe_support import pipeline as support
from taxosafe_domain import backend, core, importer, protocol, runner


class DomainLifecycle(discovery_fixture.DiscoveryLifecycle):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(warnings.catch_warnings())
        # The tiny fixture deliberately has one sample in some species.
        # Silence only these expected sklearn metric warnings.
        warnings.filterwarnings("ignore", message="A single label was found in", category=UserWarning)
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)

    def _domain_json(self, arm_id, stage, name="completed.json"):
        return protocol.read_json(self.domain / "arms" / arm_id / stage / name)

    def _domain_model(self, arm_id):
        return support._load_torch(self.domain / "arms" / arm_id / "training/model.pth")

    def test_twelve_real_arms_failed_gates_still_test_with_frozen_d05_and_no_refitting(self):
        discovery_fixture.DiscoveryLifecycle.test_eleven_real_arms_failed_gates_still_test_without_any_refitting(self)
        discovery = self.suite
        parent_files = reference_fixture.artifact_snapshot(discovery)
        parent = importer.load_d05(discovery)
        parent_digest = backend._semantic(parent.payload)
        cfg = protocol.validate_config(copy.deepcopy(protocol.DEFAULTS))
        self.domain = self.root / "domain"
        calls, test_reads = [], []
        original_load, original_read = support._load_torch, Path.read_text
        real_bank_fit = core.DomainBank.fit

        def guarded_load(path):
            if Path(path).resolve() == (discovery / "cache/test/features.pth").resolve():
                self.assertTrue((self.domain / "dev_selection.json").is_file())
                test_reads.append(str(path))
            return original_load(path)

        def no_old_test_predictions(path, *args, **kwargs):
            try:
                relative = Path(path).resolve().relative_to(discovery.resolve())
            except ValueError:
                relative = None
            if relative is not None and "test" in relative.parts and relative.name in ("predictions.jsonl", "scores.jsonl"):
                raise AssertionError("Domain read historical TEST predictions")
            return original_read(path, *args, **kwargs)

        def in_process_worker(suite, arm_id, stage, device):
            calls.append((arm_id, stage))
            logs = suite / "logs" / (arm_id or "cache")
            logs.mkdir(parents=True, exist_ok=True)
            stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
            stdout.write_text("tiny real Domain integration\n")
            try:
                with ExitStack() as stack:
                    if stage in ("cache_test", "test"):
                        self.assertTrue((suite / "dev_selection.json").is_file())
                        for target in (
                            "taxosafe_domain.core.DomainBank.fit",
                            "taxosafe_domain.core._fit_models",
                            "taxosafe_domain.core._fit_model",
                            "taxosafe_domain.core._normalizer",
                            "taxosafe_domain.core._eigh",
                            "taxosafe_domain.calibration.fit_router",
                            "taxosafe_domain.calibration.crossfit_audit",
                            "taxosafe_boundary.core.BoundaryBank.fit",
                            "taxosafe_boundary.core.BoundaryVerifier.fit",
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

        with legacy_torch_apis(isin_mode="missing"), \
                patch.object(torch.linalg, "eigh", new=None), \
                patch.object(support, "_load_torch", side_effect=guarded_load), \
                patch.object(Path, "read_text", new=no_old_test_predictions), \
                patch.object(support, "make_loader", side_effect=AssertionError("Domain must reuse frozen features without image forwarding")), \
                patch.object(core.DomainBank, "fit", wraps=real_bank_fit) as fitted, \
                patch.object(runner, "_launch_stage", side_effect=in_process_worker), \
                patch("builtins.print"):
            preflight = runner.preflight(cfg, discovery, self.domain)
            self.assertFalse(self.domain.exists())
            self.assertFalse(preflight["test_cache_opened"])
            self.assertEqual(test_reads, [])
            summary = runner.execute_suite(cfg, discovery, self.domain)
            self.assertEqual(summary["completed_calibration_count"], 12, summary["technical_failures"])
            self.assertEqual(summary["completed_test_count"], 12, summary["technical_failures"])
            self.assertEqual(summary["technical_failures"], {})
            self.assertEqual(len(summary["experiment_matrix"]), 12)
            self.assertEqual(fitted.call_count, 2, "Only H04 main and H11 wide fit statistics")
            self.assertTrue(all(not row["dev_targets_passed"] for row in summary["all_arms"]))
            self.assertEqual(summary["recommendation_arm_id"], "H00_reference")
            index = calls.index((None, "cache_test"))
            self.assertEqual(sum(stage == "calibration" for _, stage in calls[:index]), 12)
            self.assertTrue(all(stage == "test" for _, stage in calls[index + 1:]))
            self.assertGreater(len(test_reads), 0)
            self.assertEqual(self._domain_json("H00_reference", "calibration", "router.json"), parent.info["reference"]["router"])
            self.assertEqual(self._domain_json("H01_d05", "calibration", "router.json"), parent.router)
            reproduction = self._domain_json("H01_d05", "calibration", "report.json")["calibration_diagnostics"]["source_reproduction"]
            self.assertTrue(all(reproduction[k] for k in ("exact_scores", "exact_candidates", "exact_terminal_outputs")))
            main_receipt = self._domain_json("H04_subspace_root", "training")
            inherited_reference = self._domain_json("H00_reference", "training")["fit_report"]
            inherited_d05 = self._domain_json("H01_d05", "training")["fit_report"]
            self.assertEqual(inherited_reference["initialized_from"], "original_reference")
            self.assertEqual(inherited_reference["source_model_sha256"],
                             parent.info["reference"]["training"]["checkpoint"]["sha256"])
            self.assertEqual(inherited_d05["initialized_from"], "original_D05")
            self.assertEqual(inherited_d05["source_model_sha256"], parent.info["training"]["model"]["sha256"])
            for arm in cfg["arms"]:
                arm_id = arm["id"]
                self.assertEqual(self._domain_json(arm_id, "training")["optimizer_steps"], 0)
                payload = self._domain_model(arm_id)
                self.assertEqual(backend._semantic(payload["parent_payload"]), parent_digest)
                if arm["kind"] == "reuse":
                    self.assertEqual(self._domain_json(arm_id, "training")["model"], main_receipt["model"])
            wide_receipt = self._domain_json("H11_dual_rank16", "training")
            self.assertNotEqual(main_receipt["model"]["sha256"], wide_receipt["model"]["sha256"])
            selection_digest = protocol.file_hash(self.domain / "dev_selection.json")
            freeze = datetime.fromisoformat(summary["dev_selection"]["created_at_utc"])
            for arm in cfg["arms"]:
                arm_id = arm["id"]
                receipt = self._domain_json(arm_id, "test")
                self.assertFalse(receipt["calibration_gate_is_execution_gate"])
                self.assertTrue(receipt["test_allowed_after_failed_gates"])
                self.assertEqual(receipt["dev_selection_sha256"], selection_digest)
                dev_receipt = self._domain_json(arm_id, "calibration")
                test_report = self._domain_json(arm_id, "test", "report.json")
                self.assertEqual(receipt["inherited_dev_targets_passed"], dev_receipt["targets_passed"])
                self.assertEqual(test_report["calibration_status"], "passed" if dev_receipt["targets_passed"] else "best_effort")
                self.assertEqual(test_report["evaluation_status"], "passed" if receipt["targets_passed"] else "best_effort")
                self.assertEqual(test_report["calibration_status_origin"], "frozen_development")
                records = reference_fixture.read_records(self.domain / "arms" / arm_id / "test/predictions.jsonl")
                self.assertEqual(len(records), 11)
                self.assertEqual(sum(row["evaluation_weight"] for row in records), 10)
                for stage in ("training", "calibration", "test"):
                    marker = self._domain_json(arm_id, stage, "stage_binding.json")
                    self.assertGreaterEqual(marker["elapsed_seconds"], 0)
                    if stage == "test":
                        self.assertGreaterEqual(datetime.fromisoformat(marker["started_at_utc"]), freeze)
            self._assert_domain_decisions(cfg)
            self._assert_truth_independent_inference(cfg)
            original_calls = list(calls)
            runner.execute_suite(cfg, discovery, self.domain, resume=True)
            self.assertEqual(calls, original_calls)
            self.assertEqual(protocol.file_hash(self.domain / "dev_selection.json"), selection_digest)
            changed = copy.deepcopy(cfg)
            changed["name"] += " modified"
            with self.assertRaises(ValueError):
                runner.execute_suite(changed, discovery, self.domain, resume=True)
            with patch.object(protocol, "code_signature", return_value="changed-runtime-code"):
                with self.assertRaises(ValueError):
                    runner.execute_suite(cfg, discovery, self.domain, resume=True)
            self.assertEqual(calls, original_calls)
            path = self.domain / "arms/H08_dual_conditional/calibration/router.json"
            original = path.read_bytes()
            try:
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    runner.execute_suite(cfg, discovery, self.domain, resume=True)
            finally:
                path.write_bytes(original)
            self.assertEqual(calls, original_calls)
        self.assertEqual(backend._semantic(parent.payload), parent_digest)
        self.assertEqual(reference_fixture.artifact_snapshot(discovery), parent_files)
        # Historical predictions are read only by this post-run assertion;
        # runtime was explicitly forbidden from reading them throughout.
        for old_arm, new_arm in (("D00_reference", "H00_reference"), ("D05_episode_bce", "H01_d05")):
            for phase in ("calibration", "test"):
                old = reference_fixture.read_records(discovery / "arms" / old_arm / phase / "predictions.jsonl")
                new = reference_fixture.read_records(self.domain / "arms" / new_arm / phase / "predictions.jsonl")
                old = {r["image_sha256"]: r for r in old}
                new = {r["image_sha256"]: r for r in new}
                self.assertEqual(set(old), set(new))
                for identity in old:
                    keys = ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf")
                    for key in keys:
                        self.assertEqual(old[identity][key], new[identity][key])
                    if old_arm == "D05_episode_bce":
                        self.assertEqual(old[identity]["discovery"], new[identity]["discovery"])
        self._assert_source_unchanged()

    def _assert_domain_decisions(self, cfg):
        for phase in ("calibration", "test"):
            rows = {}
            for arm in cfg["arms"]:
                records = reference_fixture.read_records(self.domain / "arms" / arm["id"] / phase / "predictions.jsonl")
                rows[arm["id"]] = {r["image_sha256"]: r for r in records}
            root_sets = [{h for h, r in rows[arm].items() if r["prediction_type"] == "global_unknown"}
                         for arm in ("H06_dual_root", "H08_dual_conditional", "H09_dual_reroute")]
            self.assertEqual(root_sets[0], root_sets[1])
            self.assertEqual(root_sets[0], root_sets[2])
            shared = ("H06_dual_root", "H08_dual_conditional", "H09_dual_reroute")
            routers = [self._domain_json(a, phase, "router.json") for a in shared]
            for key in ("root_threshold", "root_state", "root_state_sha256"):
                self.assertEqual(routers[0][key], routers[1][key])
                self.assertEqual(routers[0][key], routers[2][key])
            for identity in rows[shared[0]]:
                for other in shared[1:]:
                    left, right = rows[shared[0]][identity], rows[other][identity]
                    self.assertEqual(left["domain"]["root_score"], right["domain"]["root_score"])
                    self.assertEqual(left["root_pass"], right["root_pass"])
            anchors = rows["H01_d05"]
            for arm in cfg["arms"][2:]:
                router = self._domain_json(arm["id"], phase, "router.json")
                for identity, row in rows[arm["id"]].items():
                    self.assertEqual(row["domain"]["anchor_parent"], anchors[identity]["candidate_parent"])
                    self.assertEqual(row["domain"]["anchor_leaf"], anchors[identity]["candidate_leaf"])
                    self.assertEqual(row["root_pass"], row["domain"]["root_score"] >= router["root_threshold"])
                    self.assertEqual(row["prediction_type"] == "global_unknown", not row["root_pass"])
                    if arm["candidate_policy"] == "reference_path":
                        for key in ("candidate_parent", "candidate_leaf"):
                            self.assertEqual(row[key], anchors[identity][key])

    def _assert_truth_independent_inference(self, cfg):
        _, _, info = backend._checked(self.domain)
        cache, _ = backend._load_cache(self.domain, "test", cfg, info)
        altered = copy.deepcopy(cache)
        mapping = info["meta"]["leaf_to_parent"]
        for group in altered["groups"].values():
            for row in group["records"]:
                row["true_leaf"] = ((row.get("true_leaf") or 0) + 1) % len(mapping)
                row["true_parent"] = mapping[row["true_leaf"]]
                row["status"] = "extra" if row["status"] != "extra" else "known"
        with patch.object(core.DomainBank, "fit", side_effect=AssertionError("Inference refit")):
            for arm in cfg["arms"][2:]:
                payload = self._domain_model(arm["id"])
                before = backend.score_groups(cache, arm, payload, info["meta"])
                after = backend.score_groups(altered, arm, payload, info["meta"])
                for split in before:
                    self.assertEqual([r["domain"] for r in before[split]], [r["domain"] for r in after[split]])


for _name in dir(discovery_fixture.DiscoveryLifecycle):
    if _name.startswith("test_") and _name not in DomainLifecycle.__dict__:
        setattr(DomainLifecycle, _name, None)


if __name__ == "__main__":
    unittest.main()
