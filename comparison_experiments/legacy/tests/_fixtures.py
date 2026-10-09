"""Synthetic plan inputs; unit tests must not depend on a research dataset."""
from pathlib import Path

import yaml

from tools import run_taxosafe_suite as suite


def synthetic_suite_config(root, name="SYNTHETIC_PROTOCOL_TEST"):
    root = Path(root)
    cfg = yaml.safe_load(suite.resolve(suite.DEFAULT_CONFIG).read_text())
    cfg["data"]["name"] = name
    cfg["exp"] = "unit-test"
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for key in ("train", "val_known", "val_intra", "val_extra",
                "test_known", "test_intra", "test_extra", "oe_train"):
        path = inputs / (key + ".txt")
        path.write_text("synthetic/{}.jpg,0,0\n".format(key), encoding="utf-8")
        cfg["data"][key] = str(path)
    hierarchy = inputs / "tree.npy"
    hierarchy.write_bytes(b"SYNTHETIC HIERARCHY; NOT A NUMPY ARRAY")
    cfg["data"]["hierarchy"] = str(hierarchy)
    return cfg


def freeze_legacy_oe_input(plan, cfg):
    """Model the complete-input receipt expected by historical follow-up tools.

    Old receipts may contain an OE hash even when lambda_oe is zero. The current
    suite omits that unused input, so add it explicitly only when exercising
    those older receipts; never enable OE or weaken source validation.
    """
    path = suite.resolve(cfg["data"]["oe_train"])
    plan["inputs_sha256"][suite.portable(path)] = suite.file_hash(path)
