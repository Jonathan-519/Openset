"""Real PNG decoding, production preprocessing and MaPLe spatial extraction."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from models.maple_model import VisionTransformer_MaPLe
from taxosafe_support.encoders import SupportEncoder
from taxosafe_support import pipeline as support
from taxosafe_morphology import features, protocol


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.image_encoder=VisionTransformer_MaPLe(224,16,64,1,1,8,
            dict(trainer="MaPLe",maple_length=2))
        self.prompt_learner=nn.Parameter(torch.zeros(2,64))

    def encode_image_with_spatial(self,images,normalize=True):
        glob,spatial=self.model.image_encoder.forward_with_spatial(images,self.prompt_learner,[])
        return F.normalize(glob,dim=-1),F.normalize(spatial,dim=-1)


class SpatialImages(unittest.TestCase):
    def test_raw_pngs_produce_196_reference_patches_and_detect_wrong_inputs(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,previous)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            generator=np.random.RandomState(7)
            for i,shape in enumerate(((41,79,3),(88,33,3))):
                path=root/(str(i)+".png")
                Image.fromarray(generator.randint(0,256,size=shape,dtype=np.uint8)).save(path)
                rows.append(dict(resolved_path=str(path),path=path.name,image_sha256=protocol.file_hash(path),
                                 status="known",split="train",source="a",true_leaf=0,true_parent=0))
            meta=dict(leaf_names=["a"],parent_names=["P"],leaf_to_parent=[0])
            cfg=dict(data=dict(data_root=str(root),resize_mode="letterbox",batch_size=2,eval_batch_size=2,n_workers=0))
            encoder=SupportEncoder(ImageBackbone(),meta,dict(adapter_dim=4,local_tokens=2)).eval().requires_grad_(False)
            values=[]
            for images,_,_ in support.make_loader(rows,cfg,meta,False):
                with torch.no_grad():values.append(encoder.encode(images,classify=False))
            group=dict(records=rows,image_sha256=[r["image_sha256"] for r in rows],
                       features={"source_"+key:torch.cat([v[key] for v in values]) for key in ("fine","parent")})
            cache=dict(groups=dict(train=group))
            reference=SimpleNamespace(encoder=encoder,binding=dict(fixture="real PNG and MaPLe"),config=cfg,meta=meta)
            info=dict(reference=dict(directory=root,binding=reference.binding,config=cfg))
            settings=dict(protocol.DEFAULTS["features"],batch_size=2)
            with patch.object(features,"load_reference",return_value=reference),patch("builtins.print"):
                spatial=features.collect_spatial(cache,info,settings)
                features.validate_spatial(spatial,cache,info,settings)
                self.assertEqual(list(spatial["groups"]["train"]["tokens"].shape),[2,196,8])
                self.assertEqual(list(spatial["groups"]["train"]["positions"].shape),[196,2])
                self.assertEqual(spatial["groups"]["train"]["reference_feature_max_abs_gap"],0.)
                corrupted=copy.deepcopy(cache);corrupted["groups"]["train"]["features"]["source_fine"]+=.1
                with self.assertRaises(ValueError):features.collect_spatial(corrupted,info,settings)
                spatial["groups"]["train"]["tokens"][0,0,0]+=.1
                with self.assertRaises(ValueError):features.validate_spatial(spatial,cache,info,settings)


if __name__=="__main__":
    unittest.main()
