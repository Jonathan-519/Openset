"""Synthetic mechanism/integration tests; no biological performance claims."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from taxosafe_witness import core, calibration, pipeline
from taxosafe_visual.runtime import write_json, sha256
from tools import run_taxosafe_witness_v5 as runner


def settings():
    s = yaml.safe_load((runner.ROOT / runner.CONFIG).read_text())
    s.update(projection_dim=8, support_per_leaf=6, query_per_leaf=6, regularizers=[.1, 1.])
    return s


def fixture():
    meta = {"parent_names": ["A", "B"], "leaf_names": ["a", "b", "c", "d"], "leaf_to_parent": [0, 0, 0, 1]}
    rng = np.random.RandomState(74)
    centers = core.unit(np.eye(10)[:4] + .25)
    def make(split):
        rows, gs, tokens = [], [], []
        statuses = ["known"] if split == "train" else ["known", "intra", "extra"]
        for status in statuses:
            for c in range(4 if status == "known" else 2):
                for k in range(12 if split == "train" else 6):
                    tag = "{}_{}_{}_{}".format(split, status, c, k)
                    parent = meta["leaf_to_parent"][c] if status == "known" else c if status == "intra" else None
                    center = centers[c] if status == "known" else core.unit(np.eye(10)[4 + c] + .25)
                    g = core.unit(center + rng.normal(0, .08, 10))
                    gs.append(g); tokens.append(core.unit(g + rng.normal(0, .15, (4, 10))))
                    rows.append({"status": status, "split": "train" if split == "train" else split + "_" + status,
                                 "true_leaf": c if status == "known" else None, "true_parent": parent,
                                 "image_sha256": hashlib.sha256(tag.encode()).hexdigest(), "path": tag,
                                 "source": status + str(c)})
        return {"global": np.asarray(gs), "patches": np.asarray(tokens)}, rows
    return meta, make


def anchored(data, rows, train, train_rows, meta, s):
    p = np.asarray([r["true_parent"] if r["true_parent"] is not None else 0 for r in rows])
    # Synthetic root/encoder only; actual verifier and policy execute unmocked.
    e = core.evidence(core.prepare(data, s), core.prepare(train, s),
                      [r["true_leaf"] for r in train_rows], meta["leaf_to_parent"], p, s)
    anchor = []
    for i, row in enumerate(rows):
        leaf, parent = int(e["leaf"][i]), int(p[i])
        record = dict(row, candidate_parent=parent, candidate_leaf=leaf, pred_leaf=leaf, pred_parent=parent,
                      parent=parent, leaf=leaf, prediction_type="known", text_pred_parent=parent,
                      global_pred_leaf=leaf, root_knownness_score=1., root_gate_margin=1.,
                      child_knownness_score=float(-e["x"][i, 0]), child_evidence={},
                      candidate_leaf_name=meta["leaf_names"][leaf], candidate_parent_name=meta["parent_names"][parent])
        anchor.append(record)
    return rows, anchor, e


class WitnessTests(unittest.TestCase):
    def test_collage_cannot_masquerade_as_one_coherent_reference(self):
        a, b = np.eye(2)
        bank = {"global": core.unit([[1, .1], [1, .2]]),
                "patches": np.asarray([[a, a], [b, b]]), "weights": np.ones((2, 2)) / 2}
        q = {"global": core.unit([[1, .15]]), "patches": np.asarray([[a, b]]), "weights": np.ones((1, 2)) / 2}
        e = core.evidence(q, bank, [0, 0], [0], [0], settings())
        self.assertGreater(e["x"][0, 4], .4)
        q["patches"][0] = [a, a]
        coherent = core.evidence(q, bank, [0, 0], [0], [0], settings())
        self.assertAlmostEqual(coherent["x"][0, 4], 0.)
        self.assertFalse(e["valid"][0, 5])

    def test_patch_order_and_query_chunk_do_not_change_evidence(self):
        meta, make = fixture(); d, r = make("train"); q, qr = make("val")
        s = settings(); p = np.zeros(len(qr), int)
        bank, query = core.prepare(d, s), core.prepare(q, s)
        y = np.asarray([x["true_leaf"] for x in r])
        whole = core.evidence(query, bank, y, meta["leaf_to_parent"], p, s)
        pieces = [core.evidence(core.subset(query, ids), bank, y, meta["leaf_to_parent"], p[ids], s)["x"]
                  for ids in np.array_split(np.arange(len(qr)), 3)]
        np.testing.assert_allclose(whole["x"], np.concatenate(pieces), atol=1e-7)
        query["patches"] = query["patches"][:, ::-1]; query["weights"] = query["weights"][:, ::-1]
        flipped = core.evidence(query, bank, y, meta["leaf_to_parent"], p, s)
        np.testing.assert_allclose(whole["x"], flipped["x"], atol=1e-6)

    def test_fixed_candidate_and_single_reference_missing_features(self):
        meta, make = fixture(); d, r = make("train"); s = settings(); data = core.prepare(d, s)
        refs = np.array([0, 12]); q = core.subset(data, [1])
        e = core.evidence(q, core.subset(data, refs), [0, 1], meta["leaf_to_parent"], [0], s, candidates=[1])
        self.assertEqual(e["leaf"].tolist(), [1])
        self.assertFalse(e["valid"][0, 4]); self.assertFalse(e["valid"][0, 6])
        with self.assertRaises(ValueError):
            core.evidence(q, core.subset(data, refs), [0, 1], meta["leaf_to_parent"], [1], s)

    def test_train_split_disjoint_deterministic_and_rejects_duplicate(self):
        _, make = fixture(); _, rows = make("train")
        a = core.split_train(rows, settings(), 1); b = core.split_train(rows, settings(), 1)
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)
        self.assertFalse(set(a[0]) & set(a[1])); self.assertFalse(set(a[0]) & set(a[2])); self.assertFalse(set(a[1]) & set(a[2]))
        rows[1]["image_sha256"] = rows[0]["image_sha256"]
        with self.assertRaises(ValueError): core.split_train(rows, settings(), 1)

    def test_outer_species_absent_from_all_fit_evidence_and_moments(self):
        meta, make = fixture(); d, rows = make("train"); s = settings()
        support, fit, _ = core.split_train(rows, s, 1)
        y = np.asarray([r["true_leaf"] for r in rows])
        world = core.training_world(core.prepare(d, s), y, meta["leaf_to_parent"], support, fit, s, 0)
        altered = copy.deepcopy(d)
        altered["global"][y == 0] = np.roll(altered["global"][y == 0], 3, axis=-1)
        altered["patches"][y == 0] *= -1
        other = core.training_world(core.prepare(altered, s), y, meta["leaf_to_parent"], support, fit, s, 0)
        np.testing.assert_array_equal(world["x"], other["x"])
        self.assertTrue(all(y[i] != 0 for i in world["reference_ids"] + world["fit_query_ids"]))
        self.assertEqual(core.fit_verifier(world, list(range(7)), .1), core.fit_verifier(other, list(range(7)), .1))

    def test_monotone_verifier_serializes_and_missing_is_neutral(self):
        meta, make = fixture(); d, rows = make("train"); s = settings()
        support, fit, _ = core.split_train(rows, s, 1)
        y = np.asarray([r["true_leaf"] for r in rows])
        w = core.training_world(core.prepare(d, s), y, meta["leaf_to_parent"], support, fit, s)
        model = core.fit_verifier(w, list(range(7)), .1)
        a = core.score(model, w)
        changed = dict(w, x=w["x"] + .2)
        self.assertTrue(np.all(core.score(model, changed) <= a + 1e-12))
        np.testing.assert_array_equal(a, core.score(json.loads(json.dumps(model)), w))
        changed = copy.deepcopy(w); changed["x"][~changed["valid"]] = 1e8
        np.testing.assert_array_equal(a, core.score(model, changed))

    def test_whole_species_selection_train_only_and_provenance(self):
        meta, make = fixture(); d, rows = make("train")
        models, report = core.fit_all(d, rows, meta, settings(), 1)
        self.assertEqual(set(models), set(core.VARIANTS))
        self.assertEqual(len(report["outer_folds"]), 3)
        for fold in report["outer_folds"]:
            self.assertNotIn(fold["held_leaf"], fold["fit_leaves"])
            self.assertFalse(set(fold["audit_query_ids"]) & set(fold["fit_query_ids"]))
        self.assertFalse(report["validation_loaded"]); self.assertFalse(report["classifier_updated"])
        rows[0]["split"] = "val_known"
        with self.assertRaises(ValueError): core.fit_all(d, rows, meta, settings(), 1)

    def test_known_only_calibration_does_not_use_unknown_or_audit_scores(self):
        meta, make = fixture(); d, r = make("train"); q, qr = make("val"); s = settings()
        rows, anchor, e = anchored(q, qr, d, r, meta, s)
        ids, audit = calibration.partition(rows, list(range(len(rows))), .5, 1)
        self.assertTrue(all(rows[i]["status"] == "known" for i in ids))
        self.assertFalse(set(ids) & set(audit))
        scores = -e["x"][:, 0]
        tau, _ = calibration.reference_thresholds(scores, rows, anchor, ids, 2, .9)
        ref = np.array([scores[i] >= tau[str(x["candidate_parent"])] for i, x in enumerate(anchor)])
        a = calibration.matched_thresholds(scores, rows, anchor, ids, ref, 2, .9)
        changed = scores.copy(); changed[audit] = np.arange(len(audit)) * 100
        b = calibration.matched_thresholds(changed, rows, anchor, ids, ref, 2, .9)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError): calibration.reference_thresholds(scores, rows, anchor, audit, 2, .9)

    def test_calibration_missing_parent_falls_back_and_ties_are_inclusive(self):
        meta, make = fixture(); d, r = make("train"); q, qr = make("val"); s = settings()
        rows, anchor, e = anchored(q, qr, d, r, meta, s)
        ids = [i for i, x in enumerate(rows) if x["status"] == "known" and x["true_parent"] == 0]
        scores = np.arange(len(rows), dtype=float)
        tau, reports = calibration.reference_thresholds(scores, rows, anchor, ids, 2, .9)
        self.assertEqual(reports["1"]["fallback"], "pooled_known")
        result = pipeline.apply({"taxonomy": meta}, anchor, e, scores, tau, "identity")
        for i in ids:
            if scores[i] == tau['0']:
                self.assertEqual(result[i]["prediction_type"], "known")

    def test_policy_keeps_every_candidate_and_ignores_query_truth(self):
        meta, make = fixture(); d, r = make("train"); q, qr = make("val"); s = settings()
        rows, anchor, e = anchored(q, qr, d, r, meta, s)
        anchor[-1]["prediction_type"] = "global_unknown"
        a = pipeline.apply({"taxonomy": meta}, anchor, e, -e["x"][:, 0], {'0': 1., '1': 1.}, "full")
        altered = copy.deepcopy(anchor)
        for r in altered: r.update(true_leaf=999, true_parent=999, status="extra", source="unknown")
        b = pipeline.apply({"taxonomy": meta}, altered, e, -e["x"][:, 0], {'0': 1., '1': 1.}, "full")
        for before, x, y in zip(anchor, a, b):
            for key in ("candidate_leaf", "candidate_parent", "root_gate_margin", "root_knownness_score"):
                self.assertEqual(before[key], x[key]); self.assertEqual(x[key], y[key])
            self.assertEqual(x["prediction_type"], y["prediction_type"])
        self.assertEqual(a[-1]["prediction_type"], "global_unknown")

    def test_actual_fit_calibrate_test_with_synthetic_frozen_encoder(self):
        meta, make = fixture(); d, rows = make("train"); q, qr = make("val"); td, tr = make("test"); s = settings()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); src = folder / 'source.json'; write_json(src, {"suite": str(folder), "taxonomy": meta})
            plan = {"source_plan": str(src), "suite": str(folder), "taxonomy": meta, "settings": s, "seed": 1}
            write_json(folder/'artifacts/identity/calibration.json', {'profiles': {p: {'0': 1., '1': 1.}
                       for p in ('coverage', 'balanced', 'protected')}})
            with patch.object(pipeline.old, 'load_cache', return_value=(d, rows)):
                pipeline.fit(plan)
            bundles = {'val': anchored(q, qr, d, rows, meta, s), 'test': anchored(td, tr, d, rows, meta, s)}
            with patch.object(pipeline, 'anchored_evidence', side_effect=lambda p, name: bundles[name]), patch.object(pipeline.old, 'validation_indices', return_value=list(range(len(qr)))):
                pipeline.calibrate(plan); pipeline.test(plan)
            val = json.loads((folder/'validation_report.json').read_text())
            out = json.loads((folder/'summary.json').read_text())
            self.assertFalse(val['test_loaded']); self.assertFalse(val['child_thresholds_use_unknowns'])
            self.assertEqual(len(out['rows']), 5)
            self.assertEqual(len({r['closed_routed_accuracy'] for r in out['rows']}), 1)
            self.assertEqual(len({r['extra_far'] for r in out['rows']}), 1)
            self.assertTrue(out['classification_predictions_identical'])

    def test_receipts_reject_mutated_output_and_partial_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); plan = {"suite": str(folder), "inputs_sha256": {}, "settings": {"variants": list(core.VARIANTS)}}
            write_json(folder/'plan.json', plan)
            for name in runner.outputs(plan, 'fit'): write_json(folder/name, {})
            with self.assertRaises(FileExistsError): runner.run(plan, 'fit', resume=True)
            write_json(folder/'receipts/fit.json', {'plan_sha256': sha256(folder/'plan.json'),
                'outputs': {n: sha256(folder/n) for n in runner.outputs(plan, 'fit')}})
            self.assertTrue(runner.verified(plan, 'fit'))
            write_json(folder/'models.json', {'changed': True})
            with self.assertRaises(ValueError): runner.verified(plan, 'fit')

    def test_plan_preserves_old_files_and_development_does_not_open_test(self):
        s = settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old_dir = root/'old/seed_1'; old_dir.mkdir(parents=True)
            config = root/'config.yml'; config.write_text(yaml.safe_dump(s))
            frozen = root/'upstream.py'; frozen.write_text('unchanged = True\n')
            src = {'suite': str(old_dir), 'seed': 1, 'taxonomy': fixture()[0],
                   'inputs_sha256': {str(frozen): sha256(frozen)}}
            write_json(old_dir/'plan.json', src)
            for stage in ('cache_train', 'train', 'cache_val', 'calibrate'):
                write_json(old_dir/'receipts'/ (stage+'.json'), {})
                write_json(old_dir/(stage+'.json'), {})
            before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(runner.old_runner, 'verify_inputs'), patch.object(runner.old_runner, 'verified', return_value=True), patch.object(runner.old_runner, 'outputs', side_effect=lambda p, stage: [stage+'.json']):
                path = runner.make_plan(root/'old', root/'new', 1, config)
            plan = json.loads(path.read_text())
            self.assertFalse(plan['test_loaded'])
            self.assertFalse(any('cache_test' in p or '/test/' in p for p in plan['inputs_sha256']))
            for p, digest in before.items(): self.assertEqual(sha256(p), digest)
            with patch.object(runner, 'test_inputs', side_effect=AssertionError('test was read')):
                runner.run(plan, 'develop', dry_run=True)

    def test_guard_reports_known_accuracy_tradeoff_instead_of_success(self):
        ref = {'closed_routed_accuracy': .9, 'known_end_to_end_leaf_accuracy': .8,
               'known_correct_acceptance_macro': .8, 'intra_oser': .2,
               'intra_macro_parent_species_auroc': .8, 'extra_final_known_false_acceptance': .1}
        full = dict(ref, known_end_to_end_leaf_accuracy=.7, intra_oser=.05)
        gate = pipeline.gate(ref, full)
        self.assertFalse(gate['known_correct_acceptance_not_lower'])
        self.assertFalse(gate['all_observed_checks_pass'])


if __name__ == '__main__':
    unittest.main()
