"""Synthetic protocol tests; these are not biological performance results."""
import copy, hashlib, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np, yaml
from taxosafe_dual import core, pipeline
from taxosafe_witness import core as witness
from taxosafe_visual.runtime import write_json, sha256
from tools import run_taxosafe_dual_v7 as runner

def settings():
    x = yaml.safe_load((runner.ROOT / runner.CONFIG).read_text())
    x.update(projection_dim=8, support_per_leaf=6, query_per_leaf=6,
             regularizer=.1, conformal_alpha=.1, semantic_k=2)
    return x

def fixture():
    meta = {"parent_names": ["A", "B"], "leaf_names": ["a", "b", "c", "d"],
            "leaf_to_parent": [0, 0, 0, 1]}
    rng = np.random.RandomState(91); centers = witness.unit(np.eye(10)[:4] + .2)
    rows=[]; gs=[]; patches=[]
    for leaf in range(4):
        for index in range(12):
            tag="train_{}_{}".format(leaf,index)
            g=witness.unit(centers[leaf]+rng.normal(0,.07,10))
            gs.append(g); patches.append(witness.unit(g+rng.normal(0,.14,(4,10))))
            rows.append({"status":"known","split":"train","true_leaf":leaf,
                         "true_parent":meta["leaf_to_parent"][leaf],
                         "image_sha256":hashlib.sha256(tag.encode()).hexdigest(),
                         "path":tag,"source":str(leaf)})
    g=np.asarray(gs)
    return meta,{"global":g,"patches":np.asarray(patches),
                 "leaf_cosine":g@centers.T},rows

class DualEvidenceTests(unittest.TestCase):
    def test_declared_full_has_semantic_and_morphology_evidence(self):
        self.assertEqual(len(core.FEATURES), 11)
        self.assertEqual(core.VARIANTS["full"], list(range(11)))
        self.assertTrue(set(range(7,11)).issubset(core.VARIANTS["semantic_only"]))

    def test_candidate_conditioned_evidence_has_expected_shape(self):
        meta,data,rows=fixture(); cfg=settings(); prepared=core.prepare(data,cfg)
        support,fit,_=witness.split_train(rows,cfg,1); y=np.array([r["true_leaf"] for r in rows])
        q=fit[:5]; parent=np.array(meta["leaf_to_parent"])[y[q]]
        e=core.evidence(core.subset(prepared,q),core.subset(prepared,support),y[support],
                        meta["leaf_to_parent"],parent,cfg,candidates=y[q])
        self.assertEqual(e["x"].shape,(5,11)); self.assertEqual(e["valid"].shape,(5,11))
        np.testing.assert_array_equal(e["leaf"],y[q]); self.assertTrue(np.isfinite(e["x"]).all())

    def test_removed_leaf_is_absent_from_pseudo_unknown_candidate(self):
        meta,data,rows=fixture(); cfg=settings(); prepared=core.prepare(data,cfg)
        support,fit,_=witness.split_train(rows,cfg,2); y=np.array([r["true_leaf"] for r in rows])
        held=0; refs=support[y[support]!=held]; q=fit[y[fit]==held]
        e=core.evidence(core.subset(prepared,q),core.subset(prepared,refs),y[refs],
                        meta["leaf_to_parent"],np.zeros(len(q),int),cfg)
        self.assertFalse(np.any(e["leaf"]==held))

    def test_fit_reference_and_calibration_are_disjoint(self):
        meta,data,rows=fixture(); models,state,report=core.fit_all(data,rows,meta,settings(),1)
        s=report["split"]
        self.assertFalse(set(s["support"])&set(s["fit"])); self.assertFalse(set(s["support"])&set(s["calibration"]))
        self.assertFalse(set(s["fit"])&set(s["calibration"])); self.assertEqual(set(models),set(core.VARIANTS))
        self.assertFalse(state["validation_used"]); self.assertFalse(state["test_used"])

    def test_calibration_images_do_not_change_fitted_model(self):
        meta,data,rows=fixture(); cfg=settings(); models,state,report=core.fit_all(data,rows,meta,cfg,2)
        changed=copy.deepcopy(data); ids=report["split"]["calibration"]
        changed["global"][ids]=np.roll(changed["global"][ids],2,axis=-1)
        changed["patches"][ids]=np.roll(changed["patches"][ids],2,axis=-1)
        changed["leaf_cosine"][ids]=np.roll(changed["leaf_cosine"][ids],1,axis=-1)
        models2,state2,_=core.fit_all(changed,rows,meta,cfg,2)
        self.assertEqual(models,models2); self.assertNotEqual(state,state2)

    def test_conformal_pvalue_ties_are_conservative(self):
        state={"variants":{"full":{"parents":{"0":{"scores":[1.,2.,3.],"count":3}}}}}
        np.testing.assert_allclose(core.pvalues([0.,1.,2.5,4.],[0,0,0,0],state,"full"),[.25,.5,.75,1.])

    def test_decision_preserves_candidate_and_root(self):
        meta,data,rows=fixture(); cfg=settings(); prepared=core.prepare(data,cfg)
        support,fit,_=witness.split_train(rows,cfg,1); y=np.array([r["true_leaf"] for r in rows]); q=fit[:6]
        parents=np.array(meta["leaf_to_parent"])[y[q]]
        e=core.evidence(core.subset(prepared,q),core.subset(prepared,support),y[support],meta["leaf_to_parent"],parents,cfg,candidates=y[q])
        anchor=[]
        for j,i in enumerate(q):
            anchor.append(dict(rows[i],candidate_parent=int(parents[j]),candidate_leaf=int(y[i]),pred_parent=int(parents[j]),pred_leaf=int(y[i]),parent=int(parents[j]),leaf=int(y[i]),prediction_type="known",root_knownness_score=1.,root_gate_margin=1.,child_knownness_score=1.,child_evidence={}))
        out=pipeline.apply({"taxonomy":meta,"settings":cfg},anchor,e,np.linspace(0,1,len(q)),"full")
        for a,b in zip(anchor,out):
            self.assertEqual((a["candidate_parent"],a["candidate_leaf"],a["root_gate_margin"]),(b["candidate_parent"],b["candidate_leaf"],b["root_gate_margin"]))

    def test_runner_receipt_outputs_include_train_calibration(self):
        plan={"settings":{"variants":list(core.VARIANTS)}}
        self.assertIn("train_calibration.json",runner.outputs(plan,"fit"))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); plan.update(suite=str(folder),inputs_sha256={}); write_json(folder/"plan.json",plan)
            for name in runner.outputs(plan,"fit"): write_json(folder/name,{})
            write_json(folder/"receipts/fit.json",{"plan_sha256":sha256(folder/"plan.json"),"outputs":{n:sha256(folder/n) for n in runner.outputs(plan,"fit")}})
            self.assertTrue(runner.verified(plan,"fit"))

if __name__=="__main__": unittest.main()
