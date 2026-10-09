import copy
import hashlib
import unittest
import numpy as np
import torch
from taxosafe_reference_joint import protocol,memory,training,calibration
from taxosafe_reference_joint.model import JointModel

META=dict(leaf_names=["a","b","c","d"],parent_names=["p","q"],leaf_to_parent=[0,0,1,1])


class JointCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.labels=torch.arange(4).repeat_interleave(6)
        self.fine=torch.nn.functional.normalize(torch.randn(24,8),dim=-1)
        self.parent=self.fine.clone()
        self.hashes=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(24)]
        self.mem=memory.build_memory(self.fine,self.parent,self.labels,self.hashes,META,dict(modes=2,folds=3),1)

    def test_all_held_images_excluded_and_support_is_real(self):
        for bank in self.mem["banks"][1:]:
            held={self.hashes[i] for i,f in enumerate(self.mem["fold_ids"].tolist()) if f==bank["held_fold"]}
            selected={h for row in bank["image_sha256"] for h in row}
            self.assertFalse(held&selected)
            for leaf,hashes in enumerate(bank["image_sha256"]):
                for j,h in enumerate(hashes):
                    self.assertTrue(torch.equal(bank["fine"][leaf,j],self.fine[self.hashes.index(h)]))

    def test_whole_leaf_and_parent_removal(self):
        labels=self.labels.repeat(10)
        active,target,kinds=training.episode_masks(labels,META,np.random.default_rng(4))
        self.assertEqual(set(kinds),{"known","near","extra"})
        mapping=torch.tensor(META["leaf_to_parent"])
        for y,mask,t,kind in zip(labels,active,target,kinds):
            if kind=="known":self.assertTrue(mask[y]);self.assertEqual(int(t),int(y))
            elif kind=="near":
                self.assertFalse(mask[y]);self.assertTrue(mask[mapping==mapping[y]].any());self.assertEqual(int(t),4+int(mapping[y]))
            else:
                self.assertFalse(mask[mapping==mapping[y]].any());self.assertEqual(int(t),6)

    def test_removed_support_values_cannot_change_scores(self):
        model=JointModel(8,META,protocol.ARMS[2],protocol.DEFAULTS["training"]).eval()
        bank=copy.deepcopy(self.mem["banks"][0]);text=torch.randn(6,8)
        mask=torch.ones(2,4,dtype=torch.bool);mask[:,1]=False
        a,_=model(self.fine[:2],self.parent[:2],bank,text,mask)
        bank["fine"][1]=torch.randn_like(bank["fine"][1])*100
        bank["parent"][1]=torch.randn_like(bank["parent"][1])*100
        b,_=model(self.fine[:2],self.parent[:2],bank,text,mask)
        self.assertTrue(torch.equal(a,b))

    def test_adapter_starts_as_C00_and_coarse_branch_has_no_adapter(self):
        model=JointModel(8,META,protocol.ARMS[2],protocol.DEFAULTS["training"])
        self.assertTrue(torch.allclose(model.adapt(self.fine),self.fine,atol=1e-7))
        self.assertFalse(any("coarse_adapter" in k for k in model.state_dict()))

    def test_joint_decoder_can_choose_non_C00_parent_without_truth(self):
        router=calibration._state(META,0.,0.,[])
        row=dict(joint_logits=[0.,0.,0.,0.,1.,5.,2.],true_leaf=0,true_parent=0,status="known",source="unused")
        a=calibration.decode([row],router,META)[0]
        self.assertEqual((a["prediction_type"],a["parent"]),("intra_unknown",1))
        row.update(true_leaf=None,true_parent=None,status="extra",source="different")
        b=calibration.decode([row],router,META)[0]
        self.assertEqual((a["prediction_type"],a["parent"]),(b["prediction_type"],b["parent"]))
        router["root_bias"]=10.
        with self.assertRaises(ValueError):calibration.decode([row],router,META)

    def test_virtual_samples_repeat_and_stay_finite(self):
        distribution=training._tail_distribution(self.fine,self.parent,self.labels)
        a=training._tail_samples(distribution,10,torch.Generator().manual_seed(8))
        b=training._tail_samples(distribution,10,torch.Generator().manual_seed(8))
        for x,y in zip(a,b):
            self.assertTrue(torch.equal(x,y));self.assertTrue(torch.isfinite(x).all())

    def test_config_matrix_and_TEST_calibration_are_rejected(self):
        cfg=copy.deepcopy(protocol.DEFAULTS);cfg["arms"]=cfg["arms"][:2]
        with self.assertRaises(ValueError):protocol.validate_config(cfg)
        with self.assertRaises(ValueError):calibration.fit([dict(image_sha256=self.hashes[0],split="test_known",status="known")],META,protocol.DEFAULTS["calibration"],{})


if __name__=="__main__":unittest.main()
