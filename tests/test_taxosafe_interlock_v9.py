"""Protocol and decision tests; these are not biological performance results."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from taxosafe_interlock import core, pipeline
from taxosafe_visual.runtime import write_json, sha256
from tools import run_taxosafe_interlock_v9 as runner


META = {"parent_names": ["A", "B"], "leaf_names": ["a", "b", "c"],
        "leaf_to_parent": [0, 0, 1]}
SETTINGS = {"conformal_alpha": .1, "veto_leaf_multiplier": 2.0,
            "rescue_parent_multiplier": 2.0, "rescue_leaf_multiplier": 4.0}


def record(parent=0, leaf=0, decision="known", consensus=True):
    return {"status": "known", "true_parent": parent, "true_leaf": leaf,
            "candidate_parent": parent, "candidate_leaf": leaf,
            "global_pred_leaf": leaf if consensus else (1 if leaf == 0 else 0),
            "text_pred_parent": parent if consensus else (1 - parent),
            "pred_parent": parent, "pred_leaf": leaf, "parent": parent,
            "parent_name": META["parent_names"][parent],
            "leaf": leaf if decision == "known" else None,
            "leaf_name": META["leaf_names"][leaf] if decision == "known" else None,
            "prediction_type": decision, "root_knownness_score": 1.0,
            "root_gate_margin": .5, "child_knownness_score": .5,
            "child_gate_margin": .2, "child_evidence": {"legacy": .5}}


def evidence(parents, leaves):
    n = len(parents)
    return {"parent": np.asarray(parents), "leaf": np.asarray(leaves),
            "neighbour": np.arange(n), "x": np.zeros((n, len(core.FEATURES))),
            "valid": np.ones((n, len(core.FEATURES)), dtype=bool)}


class SafetyInterlockTests(unittest.TestCase):
    def test_exact_parent_and_leaf_pvalues(self):
        state = {"variants": {"full": {"parents": {"0": {"scores": [1., 2., 3.]}},
                                              "leaves": {"0": {"scores": [1., 3.]}}}}}
        pp, lp = core.pvalue_components([0., 1., 2.5, 4.], [0]*4, [0]*4, state, "full")
        np.testing.assert_allclose(pp, [.25, .5, .75, 1.])
        np.testing.assert_allclose(lp, [1/3, 2/3, 2/3, 1.])

    def test_veto_requires_parent_leaf_and_semantic_contradiction(self):
        before = [record(decision="known", consensus=False)]
        out = pipeline.apply({"taxonomy": META, "settings": SETTINGS}, before,
                             evidence([0], [0]), [.1], [.2], "full")
        self.assertEqual(out[0]["prediction_type"], "intra_unknown")
        self.assertTrue(out[0]["interlock_vetoed"])
        for pp, lp, consensus in ((.11, .1, False), (.1, .21, False), (.1, .1, True)):
            row = record(decision="known", consensus=consensus)
            kept = pipeline.apply({"taxonomy": META, "settings": SETTINGS}, [row],
                                  evidence([0], [0]), [pp], [lp], "full")[0]
            self.assertEqual(kept["prediction_type"], "known")

    def test_leaf_can_only_rescue_with_parent_and_semantic_consensus(self):
        cases = [(record(decision="intra_unknown"), .3, .5, True),
                 (record(decision="intra_unknown"), .2, .9, False),
                 (record(decision="intra_unknown"), .9, .4, False),
                 (record(decision="intra_unknown", consensus=False), .9, .9, False)]
        for before, pp, lp, expected in cases:
            out = pipeline.apply({"taxonomy": META, "settings": SETTINGS}, [before],
                                 evidence([0], [0]), [pp], [lp], "full")[0]
            self.assertEqual(out["prediction_type"] == "known", expected)
            self.assertEqual(out["interlock_rescued"], expected)

    def test_singleton_and_global_decisions_are_preserved(self):
        singleton = record(parent=1, leaf=2, decision="known")
        global_unknown = record(parent=0, leaf=0, decision="global_unknown")
        out = pipeline.apply({"taxonomy": META, "settings": SETTINGS},
                             [singleton, global_unknown], evidence([1, 0], [2, 0]),
                             [0., 1.], [0., 1.], "full")
        self.assertEqual(out[0]["prediction_type"], "known")
        self.assertEqual(out[0]["interlock_policy"], "legacy_coverage_singleton")
        self.assertEqual(out[1]["prediction_type"], "global_unknown")

    def test_decision_never_reads_truth_and_preserves_classifier(self):
        a = record(decision="intra_unknown")
        b = copy.deepcopy(a); b.update(status="intra", true_parent=1, true_leaf=None)
        args = ({"taxonomy": META, "settings": SETTINGS}, evidence([0], [0]), [.6], [.8], "full")
        x = pipeline.apply(args[0], [a], args[1], args[2], args[3], args[4])[0]
        y = pipeline.apply(args[0], [b], args[1], args[2], args[3], args[4])[0]
        self.assertEqual(x["prediction_type"], y["prediction_type"])
        for field in ("candidate_parent", "candidate_leaf", "root_knownness_score", "root_gate_margin"):
            self.assertEqual(a[field], x[field])

    def test_runner_receipt_outputs_include_train_calibration(self):
        plan = {"settings": {"variants": list(core.VARIANTS)}}
        self.assertIn("train_calibration.json", runner.outputs(plan, "fit"))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); plan.update(suite=str(folder), inputs_sha256={})
            write_json(folder / "plan.json", plan)
            for name in runner.outputs(plan, "fit"):
                write_json(folder / name, {})
            write_json(folder / "receipts/fit.json", {"plan_sha256": sha256(folder / "plan.json"),
                       "outputs": {n: sha256(folder / n) for n in runner.outputs(plan, "fit")}})
            self.assertTrue(runner.verified(plan, "fit"))


if __name__ == "__main__":
    unittest.main()
