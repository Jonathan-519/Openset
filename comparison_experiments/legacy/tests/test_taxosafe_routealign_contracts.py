"""Boundary tests for safe immutable deployment and dependency reuse."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from taxosafe_routealign import evaluation, protocol, runner
from taxosafe_routealign.proximity import ProximityBank
from taxosafe_support import pipeline as support
from tools.pack_taxosafe_routealign_review import pack


class RouteAlignContracts(unittest.TestCase):
    def test_proximity_payload_cannot_hide_unrelated_train_bank(self):
        meta = dict(parent_names=['p'], leaf_names=['a', 'b'], leaf_to_parent=[0, 0])
        features = torch.tensor([[1., 0.], [.99, .1], [0., 1.], [.1, .99]])
        hashes = [str(i)*64 for i in range(1, 5)]
        bank = ProximityBank.fit(features, features, torch.tensor([0,0,1,1]), hashes, meta)
        source = SimpleNamespace(meta=meta, binding={'source': 'fixed'}, encoder=SimpleNamespace(dimension=2),
                                 training={'audit': {'train': {'image_hashes': hashes}}})
        binding = dict(checkpoint={'sha256': 'checkpoint'}, support={'sha256': 'support'})
        settings = {'k': 3, 'shrinkage': 10.}
        payload = evaluation._proximity_payload(source, binding, bank, settings)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'proximity.pth'
            support._save_torch(path, payload)
            evaluation._load_proximity(path, source, binding, settings)
            other = ProximityBank.fit(features, features, torch.tensor([0,0,1,1]), ['x'+h[1:] for h in hashes], meta)
            payload['bank'] = other.state_dict()
            support._save_torch(path, payload)
            with self.assertRaisesRegex(ValueError, 'internal TRAIN'):
                evaluation._load_proximity(path, source, binding, settings)

    def test_resume_rejects_orphan_reused_weight_stage(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'arms/A03_combined/calibration').mkdir(parents=True)
            (root/'arms/A03_combined/calibration'/runner.STAGE_MARKER).write_text('{}')
            arm = dict(id='A03_combined', kind='routing', weights='A01_evidence_anchor')
            with patch.object(runner, '_verify_snapshot', return_value={}), patch.object(runner, '_verify_stage', return_value={}):
                with self.assertRaisesRegex(ValueError, 'weight dependency'):
                    runner._resume_audit(root, {'arms':[arm]}, {})

    def test_pack_new_schema_preserves_source_and_excludes_weights(self):
        import tarfile
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); suite=root/'suite'; suite.mkdir()
            (suite/'snapshot.json').write_text(json.dumps({'schema_version': protocol.SCHEMA_VERSION}))
            (suite/'summary.json').write_text('{"targets_passed":false}')
            (suite/'best.pth').write_bytes(b'private weights')
            before={p.name:p.read_bytes() for p in suite.iterdir()}
            result=pack(suite,root/'review.tar.gz')
            with tarfile.open(result) as archive:
                names=archive.getnames()
                self.assertIn('suite/summary.json',names)
                self.assertNotIn('suite/best.pth',names)
            self.assertEqual(before,{p.name:p.read_bytes() for p in suite.iterdir()})

    def test_invalid_router_budget_is_rejected_before_run(self):
        cfg=protocol.effective_config(protocol.PROJECT_ROOT/protocol.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'cfg.json'
            for key in ('proximity_weight','rerank_weight','proximity_clip'):
                invalid=copy.deepcopy(cfg);invalid['router'][key]=100.
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):protocol.effective_config(path)

if __name__=='__main__':unittest.main()
