import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from taxosafe_support.protocol import PROJECT_ROOT, VARIANTS, effective_config, signature


CONFIG = PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_new.yml"


class ProtocolTest(unittest.TestCase):
    def test_all_variants_keep_existing_data_permissions_and_strict_gates(self):
        old = yaml.safe_load((CONFIG.parent / "TaxoSafe_dcbs_v11.yml").read_text())
        manifests = ("train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra")
        for variant in VARIANTS:
            cfg = effective_config(CONFIG, variant, 3)
            self.assertEqual(cfg["seed"], 3)
            for key in manifests:
                self.assertEqual(cfg["data"][key], old["data"][key])
            self.assertEqual(cfg["evaluation_gates"]["known_e2e"], {"operator": ">", "target": .9})
            self.assertNotIn("dcbs", cfg)

    def test_rejects_contaminated_initialization_and_relaxed_success_line(self):
        base = yaml.safe_load(CONFIG.read_text())
        cases = []
        cfg = copy.deepcopy(base)
        cfg["init_checkpoint"] = "v10_OE.pth"
        cases.append(cfg)
        cfg = copy.deepcopy(base)
        cfg["evaluation_gates"]["known_e2e"]["operator"] = ">="
        cases.append(cfg)
        cfg = copy.deepcopy(base)
        cfg["loss"] = {"lambda_oe": 1.0}
        cases.append(cfg)
        cfg = copy.deepcopy(base)
        cfg["support"]["max_per_leaf"] = 1
        cases.append(cfg)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yml"
            for cfg in cases:
                path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    effective_config(path)

    def test_signature_binds_new_code_and_config(self):
        cfg = effective_config(CONFIG)
        before = signature(cfg)
        self.assertIn("support_code", before)
        self.assertEqual(before["method"], "support_conditioned_v1")
        changed = copy.deepcopy(cfg)
        changed["support"]["loss"]["paired"] += .1
        self.assertNotEqual(signature(changed)["config"], before["config"])


if __name__ == "__main__":
    unittest.main()
