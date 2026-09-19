"""CPU geometry, leakage constraints, calibration ties, and end-to-end tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from boundaryshell.core import (OpenProjectionHead, fit_support, distances,
    synthesize_shell, boundary_loss, calibrate_known)
from boundaryshell.protocol import partition_known
from boundaryshell import runtime


def bank(n=40):
    torch.manual_seed(7)
    center = F.normalize(torch.eye(8)[:4] + torch.tensor([0., 0., 0., 0., 1., 0., 0., 0.]), dim=-1)
    y = torch.arange(4).repeat_interleave(n)
    x = F.normalize(center[y] + 0.025 * torch.randn(len(y), 8), dim=-1)
    return x, y, torch.tensor([0, 0, 1, 1])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x, self.y, self.mapping = bank()
        self.support = fit_support(self.x, self.y, self.mapping)

    def test_shell_outside_all_classes_and_parent_retained(self):
        u, p, c, stats = synthesize_shell(self.x, self.y, self.support)
        self.assertGreater(len(u), 0)
        self.assertTrue((distances(u, self.support['centers']) > self.support['radius']).all())
        self.assertTrue(torch.equal(p, self.mapping[c]))
        self.assertTrue(torch.allclose(u.norm(dim=-1), torch.ones(len(u)), atol=1e-5))

    def test_singleton_parent_does_not_invent_sibling(self):
        support = fit_support(self.x, self.y, torch.arange(4))
        u, p, c, stats = synthesize_shell(self.x, self.y, support)
        self.assertEqual(len(u), 0)
        self.assertEqual(stats['proposed'], 0)

    def test_knn_without_shared_parent(self):
        support = fit_support(self.x, self.y, torch.arange(4))
        u, p, _, _ = synthesize_shell(self.x, self.y, support, mode='knn')
        self.assertGreater(len(u), 0)
        self.assertTrue((p == -1).all())

    def test_gradients_update_only_open_head(self):
        head = OpenProjectionHead(8, 4, 8)
        u, p, _, _ = synthesize_shell(self.x, self.y, self.support)
        sup = fit_support(head(self.x), self.y, self.mapping)
        before = head.residual[-1].weight.detach().clone()
        opt = torch.optim.Adam(head.parameters(), lr=0.01)
        loss, _ = boundary_loss(head, self.x, self.y, u, p, sup)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before, head.residual[-1].weight))
        self.assertIsNone(self.x.grad)

    def test_shrinkage_and_missing_class(self):
        a = fit_support(self.x, self.y, self.mapping, shrinkage=0)
        self.assertTrue(torch.allclose(a['radius'], a['raw_radius'].clamp_min(0.03)))
        with self.assertRaises(ValueError):
            fit_support(self.x[self.y != 3], self.y[self.y != 3], self.mapping)

    def test_calibration_strict_90_and_ties(self):
        n = 100
        scores = torch.ones(n)
        correct = torch.arange(n) < 92
        t = calibrate_known(scores, scores, scores, correct)
        self.assertGreater(t['known_e2e'], 0.90)
        self.assertEqual(t['required_correct'], 91)
        self.assertEqual(t['coverage'], 1.)
        with self.assertRaises(ValueError):
            calibrate_known(scores, scores, scores, torch.arange(n) < 90)

    def test_root_and_local_joint_budget(self):
        n = 1000
        torch.manual_seed(2)
        local, semantic, manifold = [torch.rand(n) for _ in range(3)]
        correct = torch.arange(n) < 950
        t = calibrate_known(local, semantic, manifold, correct, root_budget=0.2)
        accepted = ~((semantic.double() < t['semantic_threshold']) & (manifold.double() < t['manifold_threshold'])) & (local.double() >= t['local_threshold'])
        self.assertGreater(float((accepted & correct).float().mean()), .90)
        self.assertGreaterEqual(int((accepted & correct).sum()), 930)

    def test_partition_stable_disjoint_exhaustive(self):
        paths = ['species/image{}.jpg'.format(i) for i in range(40)]
        labels = [i // 10 for i in range(40)]
        a, b = partition_known(paths, labels)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), list(range(40)))
        self.assertEqual((a, b), partition_known(paths, labels))


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, tag):
        self.x, self.target = x, y.tolist()
        self.data = ['{}/class{}/image{}.png'.format(tag, int(y[i]), i) for i in range(len(y))]
    def __len__(self):
        return len(self.target)
    def __getitem__(self, i):
        return self.x[i], self.target[i], i


class PipelineTests(unittest.TestCase):
    def test_open_train_calibrate_test_and_hash_binding(self):
        torch.set_num_threads(2)
        x, y, mapping = bank(20)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'classifier').mkdir()
            torch.save({}, root / 'classifier/best.pth')
            for name in ('train', 'val_known', 'hierarchy', 'test_known', 'test_intra', 'test_extra'):
                (root / name).write_text(name)
            cfg = {'data': {k: str(root / k) for k in ('train', 'val_known', 'hierarchy', 'test_known', 'test_intra', 'test_extra')},
                'boundaryshell': {'partition_seed': 1729, 'output_dim': 4, 'hidden_dim': 8, 'epochs': 2,
                    'lr': 0.001, 'batch_size': 32, 'support_quantile': 0.95, 'shrinkage': 20., 'min_radius': 0.03,
                    'neighborhood': 'taxonomy', 'neighbors': 1, 'shell_margin': 0.08, 'shell_noise': 0.15,
                    'shell_attempts': 2, 'min_synthetic': 1, 'reject_margin': 0.15, 'parent_weight': 0.2},
                'calibration': {'target': .9, 'max_drop': .02, 'root_budget': .002},
                'protocol': 'synthetic_integration_test'}
            calls, collected = [], []
            meta = {'leaf_names': ['a', 'b', 'c', 'd'], 'parent_names': ['p', 'q'], 'leaf_to_parent': mapping}
            def fake_build(config, splits, device, checkpoint=None):
                calls.append(list(splits))
                loaders = {}
                for split in splits:
                    targets = y if split not in ('test_intra', 'test_extra') else (mapping[y] if split == 'test_intra' else torch.full_like(y, -1))
                    loaders[split] = torch.utils.data.DataLoader(FakeDataset(x, targets, split), batch_size=16)
                return nn.Linear(8, 4), loaders, meta
            def fake_collect(model, loader, meta, device, status='known'):
                ids = list(loader.sampler)
                collected.append((loader.dataset.data[0].split('/')[0], ids))
                rows = []
                for i in ids:
                    leaf = int(y[i])
                    rows.append({'status': status, 'path': loader.dataset.data[i], 'dataset_index': i,
                        'image_feature': x[i].tolist(), 'global_pred_leaf': leaf, 'pred_leaf': leaf,
                        'pred_parent': int(mapping[leaf]), 'true_leaf': leaf if status == 'known' else None,
                        'true_parent': int(mapping[leaf]) if status != 'extra' else None,
                        'parent_score': .8, 'local_known_margin': .4})
                return rows
            with patch.object(runtime, 'build', fake_build), patch.object(runtime, 'collect', fake_collect):
                runtime.train_open(cfg, root, torch.device('cpu'))
                train_selection = collected[-1][1]
                self.assertFalse(any(any('test' in s for s in splits) for splits in calls))
                runtime.calibrate(cfg, root, torch.device('cpu'))
                calib_ids = collected[-1][1]
                self.assertFalse(set(train_selection) & set(calib_ids))
                runtime.test(cfg, root, torch.device('cpu'))
                metrics = json.loads((root / 'test/metrics.json').read_text())
                calibration = json.loads((root / 'calibration/thresholds.json').read_text())
                self.assertGreater(calibration['known_e2e'], .90)
                acc = metrics['known']['end_to_end_leaf_accuracy']
                closed = metrics['known']['global_leaf_accuracy']
                self.assertEqual(metrics['research_gate']['passed'],
                                 acc > .90 and closed - acc <= .02 + 1e-12)
                self.assertEqual(metrics['known']['sample_count'], len(y))
                with self.assertRaises(FileExistsError):
                    runtime.test(cfg, root, torch.device('cpu'))
                (root / 'classifier/best.pth').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    runtime.load_open(cfg, root, torch.device('cpu'))


class PreparationTests(unittest.TestCase):
    def test_clean_lists_idempotence_and_test_immutability(self):
        import csv
        from boundaryshell.protocol import prepare_known, sha256, audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = {}
            for split in ('train', 'val_known', 'test_known', 'test_intra', 'test_extra'):
                rows = []
                labels = [0, 1] if split in ('train', 'val_known', 'test_known') else ([0] if split == 'test_intra' else [-1])
                for label in labels:
                    for i in range(5 if split == 'train' else 2):
                        path = root / split / ('species' + str(label)) / (str(i) + '.png')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes((split + str(label) + str(i)).encode())
                        rows.append([str(path), label, len(rows)])
                manifests[split] = root / (split + '.txt')
                with manifests[split].open('w', newline='') as f:
                    csv.writer(f).writerows(rows)
            # One training copy of a held-out known image must be excluded.
            (root / 'train/species0/0.png').write_bytes((root / 'test_known/species0/0.png').read_bytes())
            hierarchy = root / 'tree.npy'
            hierarchy.write_bytes(b'toy tree')
            data = {s: str(p) for s, p in manifests.items()}
            cfg = {'source_known_manifests': {s: str(manifests[s]) for s in ('train', 'val_known')},
                'data': dict(data, data_root=str(root), near_test_root=str(root), ood_test_root=str(root),
                             num_known_leaves=2, hierarchy=str(hierarchy)),
                'boundaryshell': {'partition_seed': 1729}}
            for split in ('train', 'val_known'):
                cfg['data'][split] = str(root / 'clean' / (split + '.txt'))
            before = sha256(manifests['test_known'])
            report = prepare_known(cfg)
            self.assertEqual(report['counts']['train'], 9)
            self.assertEqual(prepare_known(cfg), report)
            self.assertEqual(before, sha256(manifests['test_known']))
            # This fixture uses the same species directory names for near/test;
            # preparation checks hashes, audit additionally catches species leakage.
            with self.assertRaises(ValueError):
                audit(cfg, True)


class MaPLeIntegrationTests(unittest.TestCase):
    def test_fp32_real_maple_forward_backward_without_download(self):
        # Real repository MaPLe/CLIP interfaces, reduced random transformer;
        # checks integration only, not pretrained accuracy.
        from models.maple_model import CLIP
        from models.maple import MaPLe
        from engine_taxosafe import score_label_sets
        torch.set_num_threads(2)
        design = {'trainer': 'MaPLe', 'vision_depth': 0, 'language_depth': 0,
                  'vision_ctx': 0, 'language_ctx': 0, 'maple_length': 2}
        backbone = CLIP(512, 224, 2, 768, 16, 77, 49408, 512, 8, 2, design)
        cfg = {'name': 'ViT-B/16', 'prec': 'fp32', 'n_ctx': 2,
               'ctx_init': 'a photo of a', 'prompt_depth': 2}
        with patch('models.maple.load_clip_to_cpu', return_value=backbone):
            model = MaPLe(cfg, ['a', 'b']).float()
        for name, p in model.named_parameters():
            p.requires_grad_('prompt_learner' in name)
        # Existing repository encode_text uses a hard-coded .cuda().
        with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
            scores, _, reps = score_label_sets(model, torch.randn(2, 3, 224, 224),
                                               {'leaf': ['a', 'b'], 'parent': ['p']}, True)
        loss = F.cross_entropy(scores['leaf'], torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(reps['image'].shape, (2, 512))
        self.assertEqual(reps['spatial'].shape[-1], 512)
        self.assertIsNotNone(model.model.prompt_learner.ctx.grad)
        self.assertTrue(torch.isfinite(model.model.prompt_learner.ctx.grad).all())


if __name__ == '__main__':
    unittest.main()
