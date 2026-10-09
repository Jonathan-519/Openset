"""Real temporary image/manifest tests for expanded unknown TRAIN isolation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from taxosafe_support import protocol as base
from taxosafe_evidence_guard import features


class EvidenceGuardDataContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.meta = dict(parent_names=["ParentA", "ParentB"], leaf_names=["KnownA", "KnownB"], leaf_to_parent=[0, 1])
        self.data = {key: str(self.root / key) for key in
                     ("data_root", "near_dev_root", "near_test_root", "ood_dev_root", "ood_test_root")}
        self.originals = {}
        specs = {
            "train": ("data_root", "ParentA/KnownA/train.png", 0),
            "val_known": ("data_root", "ParentB/KnownB/dev.png", 1),
            "val_intra": ("near_dev_root", "ParentA/NearDev/dev.png", 0),
            "val_extra": ("ood_dev_root", "ExtraDev/dev.png", -1),
            "test_known": ("data_root", "ParentB/KnownB/test.png", 1),
            "test_intra": ("near_test_root", "ParentB/NearTest/test.png", 1),
            "test_extra": ("ood_test_root", "ExtraTest/test.png", -1),
            "train_intra": ("near_dev_root", "ParentA/NearDev/train.png", 0),
            "oe_train": ("ood_dev_root", "ExtraDev/train.png", -1),
        }
        for index, (split, (root_key, relative, label)) in enumerate(specs.items(), 1):
            image = Path(self.data[root_key]) / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (3, 3), (index, index * 11, index * 19)).save(image)
            manifest = self.root / (split + ".txt")
            manifest.write_text("{},{},0\n".format(relative, label), encoding="utf-8")
            self.data[split] = str(manifest)
            self.originals[split] = image
        self.config = {"data": {key: value for key, value in self.data.items() if key not in features.DEFAULT_DATA}}
        groups = {name: base.read_split(self.config, name, self.meta) for name in specs if name not in features.DEFAULT_DATA}
        audit = base.audit_rows(groups)
        for split, item in audit.items():
            item["manifest_sha256"] = base.file_hash(self.data[split])
        self.info = dict(config=self.config, meta=self.meta, audit=audit,
                         training={"audit": {name: audit[name] for name in ("train", "val_known")}},
                         calibration={"audit": {name: audit[name] for name in ("val_known", "val_intra", "val_extra")}})
        self.cfg = {"data": {key: self.data[key] for key in features.DEFAULT_DATA}}

    def tearDown(self):
        self.temporary.cleanup()

    def read(self):
        return features.source_rows(self.info, "train", self.cfg)

    def test_real_unknowns_are_full_support_image_disjoint_and_overlap_dev_sources_explicit(self):
        groups, audit = self.read()
        self.assertEqual(set(groups), {"train", "train_intra", "oe_train"})
        self.assertEqual(groups["train_intra"][0]["true_parent"], 0)
        self.assertIsNone(groups["oe_train"][0]["true_parent"])
        self.assertEqual(audit["train_intra"]["development_source_overlap"], ["NearDev"])
        self.assertEqual(audit["oe_train"]["development_source_overlap"], ["ExtraDev"])
        self.assertTrue(audit["train_intra"]["full_C00_support_retained"])
        self.assertFalse(audit["train_intra"]["test_isolation"]["test_features_extracted"])
        self.assertTrue(features.audit_images(groups)["valid"])

    def test_labels_names_paths_and_config_overrides_fail_closed(self):
        cases = ("ParentA/NearDev/train.png,1,0\n", "../outside.png,0,0\n",
                 "/absolute/image.png,0,0\n", "ParentA/KnownA/train.png,0,0\n",
                 "ParentA/NearDev/train.png,0,-1\n")
        manifest = Path(self.data["train_intra"])
        original = manifest.read_text()
        for value in cases:
            with self.subTest(value=value):
                manifest.write_text(value)
                with self.assertRaises(ValueError):
                    self.read()
        manifest.write_text(original)
        Path(self.data["oe_train"]).write_text("ExtraDev/train.png,0,0\n")
        with self.assertRaisesRegex(ValueError, "label must be -1"):
            self.read()
        self.cfg["data"]["test_extra"] = "replacement.txt"
        with self.assertRaisesRegex(ValueError, "only train_intra and oe_train"):
            self.read()

    def test_aliases_of_any_known_or_heldout_image_are_rejected(self):
        target = self.originals["train_intra"]
        original = target.read_bytes()
        for split in ("train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra"):
            with self.subTest(split=split):
                target.write_bytes(self.originals[split].read_bytes())
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.read()
        target.write_bytes(original)

    def test_test_unknown_source_and_duplicate_content_are_rejected(self):
        near = self.originals["train_intra"]
        dest = near.parent.parent / "NearTest" / "other.png"
        dest.parent.mkdir()
        dest.write_bytes(near.read_bytes())
        Path(self.data["train_intra"]).write_text("ParentA/NearTest/other.png,0,0\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()
        Path(self.data["train_intra"]).write_text("ParentA/NearDev/train.png,0,0\nParentA/NearDev/train.png,0,1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.read()

    def test_symlink_escape_is_rejected_before_opening_the_image(self):
        image = self.originals["train_intra"]
        image.unlink()
        image.symlink_to(self.originals["test_extra"])
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.read()

    def test_same_unknown_source_cannot_change_its_parent_from_dev(self):
        image = Path(self.data["near_dev_root"]) / "ParentB/NearDev/train.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(self.originals["train_intra"].read_bytes())
        Path(self.data["train_intra"]).write_text("ParentB/NearDev/train.png,1,0\n")
        with self.assertRaisesRegex(ValueError, "conflicting taxonomy roles"):
            self.read()

    def test_missing_source_test_receipt_still_reads_test_identity_not_features(self):
        for name in ("test_known", "test_intra", "test_extra"):
            self.info["audit"].pop(name)
        with patch.object(features, "load_reference", side_effect=AssertionError("No encoder allowed")), \
                patch.object(features.support, "make_loader", side_effect=AssertionError("No TEST feature loader allowed")):
            _, audit = self.read()
        checked = audit["train_intra"]["test_isolation"]
        self.assertTrue(checked["test_image_hashes_read"])
        self.assertFalse(checked["test_features_extracted"])
        self.assertEqual(set(checked["manifest_sha256"]), {"test_known", "test_intra", "test_extra"})
        self.originals["train_intra"].write_bytes(self.originals["test_extra"].read_bytes())
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.read()


if __name__ == "__main__":
    unittest.main()
