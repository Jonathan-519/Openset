"""Optional CPU Torch regression tests for the opt-in training correction."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Torch is not installed; run in the ProTeCt environment")
class ConsistencyTests(unittest.TestCase):
    def fixture(self):
        import torch
        from losses.taxosafe_loss import _consistency_loss
        parent = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        leaf = torch.tensor([[1., 4., 0., 0.], [0., 0., 1., 4.]], requires_grad=True)
        meta = {"leaf_names": ["a", "b", "c", "d"], "parent_names": ["P", "Q"],
                "leaf_to_parent": torch.tensor([0, 0, 1, 1])}
        return torch, _consistency_loss, parent, leaf, meta

    def test_held_logits_have_no_gradient(self):
        torch, fn, parent, leaf, meta = self.fixture()
        fn(parent, leaf, torch.zeros(2, dtype=torch.bool), meta, [{1}, {3}]).backward()
        self.assertTrue(torch.equal(leaf.grad[:, [1, 3]], torch.zeros(2, 2)))

    def test_legacy_default_retains_held_columns(self):
        torch, fn, parent, leaf, meta = self.fixture()
        fn(parent, leaf, torch.zeros(2, dtype=torch.bool), meta).backward()
        self.assertGreater(float(leaf.grad[:, [1, 3]].abs().sum()), 0)

    def test_empty_active_branch_rejected(self):
        torch, fn, parent, leaf, meta = self.fixture()
        with self.assertRaises(ValueError):
            fn(parent, leaf, torch.zeros(2, dtype=torch.bool), meta, [{0, 1}, set()])
