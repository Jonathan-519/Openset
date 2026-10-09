"""Protocol tests using real, independently decodable tiny image fixtures."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from prepro.build_taxosieve_dataset import build, digest, json_bytes, load_seed, new_split, SPLITS


def picture(number):
    image = Image.new("RGB", (4, 4), (number % 256, number // 256, 31))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.seed_dir = self.root / "seed"
        self.seed_dir.mkdir()
        self.roots = {"known": "Known", "near_dev": "NearDev", "near_test": "NearTest",
                      "extra_dev": "OODDev", "extra_test": "OODTest"}
        self.classes = {"Known": {"Parent/Species": 0}, "NearDev": {"Parent/DevUnknown": 0},
                        "NearTest": {"Parent/TestUnknown": 0}, "OODDev": {"DevOOD": -1},
                        "OODTest": {"TestOOD": -1}}
        self.rows = []
        sequence = [("Known", "train"), ("Known", "train"), ("Known", "train"),
                    ("Known", "val_known"), ("Known", "test_known"), ("Known", "test_known"),
                    ("NearDev", "val_intra"), ("NearDev", "reserved_near"),
                    ("NearTest", "test_intra"), ("NearTest", "test_intra"),
                    ("OODDev", "val_extra"), ("OODDev", "reserved_extra"),
                    ("OODTest", "test_extra"), ("OODTest", "test_extra")]
        for number, (dirname, split) in enumerate(sequence, 1):
            self.add_baseline(dirname, split, str(number) + ".png", picture(number))
        self.taxonomy = {"tree.npy": b"frozen tree pickle bytes", "leaf_nodes.npy": b"frozen leaves",
                         "known_leaf_order.txt": b"0\tSpecies\n"}
        for name, data in self.taxonomy.items():
            (self.seed_dir / name).write_bytes(data)
        self.save_seed()

    def add_baseline(self, dirname, split, name, data):
        class_path, label = next(iter(self.classes[dirname].items()))
        path = dirname + "/" + class_path + "/" + name
        target = self.images / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row = {"path": path, "split": split, "label": label, "size": len(data),
               "blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
        self.rows.append(row)
        return path

    def save_seed(self):
        self.seed = {"schema_version": "taxosieve_seed_v1", "baseline_source_commit": "fixture",
                     "roots": self.roots, "classes": self.classes, "images": self.rows,
                     "taxonomy_sha256": {name: digest(data) for name, data in self.taxonomy.items()}}
        (self.seed_dir / "seed.json").write_bytes(json_bytes(self.seed))

    def run_build(self, **options):
        return build(self.root, "images", "out", "seed", **options)

    def inventory(self):
        return [json.loads(line) for line in (self.root / "out/inventory.jsonl").read_text().splitlines()]

    def freeze_order(self, restored=None):
        baseline = self.run_build()
        inventory = {row["path"]: row for row in self.inventory()}
        protocol = json.loads((self.root / "out/protocol.json").read_text())
        order = {"schema_version": "taxosieve_manifest_order_v1", "source_repository_commit": "fixture",
                 "baseline_inventory_sha256": baseline["inventory_sha256"], "baseline_manifest_sha256": {},
                 "splits": {}, "restored_test": restored or {"test_intra": [], "test_extra": []},
                 "relocated_source_roots": []}
        for split, meta in protocol["manifests"].items():
            rows = []
            for line in (self.root / meta["path"]).read_text().splitlines():
                path, label, _ = line.rsplit(",", 2)
                row = inventory[Path(meta["root"]).name + "/" + path]
                rows.append({"sha256": row["sha256"], "label": int(label)})
            order["splits"][split] = rows
            order["baseline_manifest_sha256"][split] = meta["sha256"]
        raw = json_bytes(order)
        (self.seed_dir / "manifest_order.json").write_bytes(raw)
        self.seed["manifest_order_sha256"] = digest(raw)
        (self.seed_dir / "seed.json").write_bytes(json_bytes(self.seed))
        return order

    def test_preserves_assignments_taxonomy_and_is_idempotent(self):
        result = self.run_build()
        self.assertEqual(result["statistics"]["inventory_count"], 14)
        self.assertTrue(result["statistics"]["all_images_accounted_for"])
        self.assertEqual({r["path"]: r["assigned_split"] for r in self.inventory()},
                         {r["path"]: r["split"] for r in self.rows})
        original = {p.name: p.read_bytes() for p in (self.root / "out").iterdir()}
        self.assertEqual(result, self.run_build())
        self.assertEqual(original, {p.name: p.read_bytes() for p in (self.root / "out").iterdir()})
        for name, data in self.taxonomy.items():
            self.assertEqual((self.root / "out" / name).read_bytes(), data)
        protocol = json.loads((self.root / "out/protocol.json").read_text())
        self.assertFalse(protocol["unknown_gradient_training"])
        self.assertFalse(protocol["manifests"]["reserved_near"]["active"])
        self.assertNotIn(str(self.root), json.dumps(protocol))
        for split in protocol["manifests"].values():
            manifest = self.root / split["path"]
            lines = manifest.read_text().splitlines()
            self.assertEqual([int(line.rsplit(",", 1)[1]) for line in lines], list(range(len(lines))))
            self.assertTrue(all((self.root / split["root"] / line.rsplit(",", 2)[0]).is_file() for line in lines))

    def test_added_deleted_and_renamed_identities_are_explicit(self):
        original_path = self.rows[0]["path"]
        renamed = str(Path(original_path).with_name("renamed.png"))
        (self.images / original_path).rename(self.images / renamed)
        deleted_path = self.rows[4]["path"]
        (self.images / deleted_path).unlink()
        new_path = "Known/Parent/Species/new.png"
        (self.images / new_path).write_bytes(picture(100))
        result = self.run_build()
        report = result["reconciliation"]
        self.assertEqual((report["added_count"], report["removed_count"], report["renamed_count"]), (1, 1, 1))
        self.assertEqual(report["removed_by_split"], {"test_known": 1})
        self.assertEqual(report["removed"][0]["path"], deleted_path)
        inventory = {r["path"]: r for r in self.inventory()}
        self.assertEqual(inventory[renamed]["assigned_split"], "train")
        self.assertEqual(inventory[renamed]["baseline_path"], original_path)
        self.assertEqual(inventory[new_path]["assigned_split"], new_split(inventory[new_path]))
        self.assertFalse(report["old_test_results_comparable_without_re_evaluation"])

    def test_modified_path_is_new_identity_and_cannot_overwrite_a_dataset(self):
        self.run_build()
        before = (self.root / "out/inventory.jsonl").read_bytes()
        (self.images / self.rows[0]["path"]).write_bytes(picture(777))
        preview = self.run_build(dry_run=True)
        self.assertEqual(preview["reconciliation"]["added_count"], 1)
        self.assertEqual(preview["reconciliation"]["removed_count"], 1)
        with self.assertRaisesRegex(ValueError, "Existing dataset differs"):
            self.run_build()
        self.assertEqual((self.root / "out/inventory.jsonl").read_bytes(), before)

    def test_test_val_train_priority_keeps_test_aliases(self):
        test_bytes = (self.images / self.rows[4]["path"]).read_bytes()
        val_bytes = (self.images / self.rows[3]["path"]).read_bytes()
        self.add_baseline("Known", "train", "train_test_duplicate.png", test_bytes)
        self.add_baseline("Known", "val_known", "val_test_duplicate.png", test_bytes)
        self.add_baseline("Known", "test_known", "test_alias.png", test_bytes)
        self.add_baseline("Known", "train", "train_val_duplicate.png", val_bytes)
        self.add_baseline("Known", "train", "train_duplicate.png", picture(1))
        self.save_seed()
        result = self.run_build()
        counts = result["statistics"]["selected_counts"]
        self.assertEqual(counts["train"], 3)
        self.assertEqual(counts["val_known"], 1)
        self.assertEqual(counts["test_known"], 3)
        self.assertEqual(result["statistics"]["selected_unique_counts"]["test_known"], 2)
        self.assertEqual(result["statistics"]["excluded_exact_duplicate_count"], 4)
        groups = {}
        for row in self.inventory():
            if row["disposition"] == "selected":
                previous = groups.setdefault(row["sha256"], row["assigned_split"])
                self.assertEqual(previous, row["assigned_split"])

    def test_duplicate_unknown_calibration_wins_over_inactive_reserve(self):
        data = (self.images / self.rows[6]["path"]).read_bytes()
        self.add_baseline("NearDev", "reserved_near", "alias.png", data)
        self.save_seed()
        result = self.run_build()
        self.assertEqual(result["statistics"]["selected_counts"]["val_intra"], 1)
        self.assertEqual(result["statistics"]["selected_counts"]["reserved_near"], 1)
        excluded = result["reconciliation"]["excluded_exact_duplicates"]
        self.assertEqual(excluded[0]["same_as_split"], "val_intra")

    def test_new_filename_alias_cannot_move_historical_train_bytes_to_test(self):
        number = next(number for number in range(100, 1000)
                      if new_split({"role": "known", "sha256": digest(picture(number))}) == "test_known")
        data = picture(number)
        prior = self.rows[0]
        (self.images / prior["path"]).write_bytes(data)
        prior["size"] = len(data)
        prior["blob_sha1"] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        self.save_seed()
        alias = "Known/Parent/Species/new_alias.png"
        (self.images / alias).write_bytes(data)
        result = self.run_build()
        inventory = {r["path"]: r for r in self.inventory()}
        self.assertEqual(inventory[alias]["assignment"], "historical_content_alias")
        self.assertEqual(inventory[alias]["assigned_split"], "train")
        self.assertEqual(inventory[prior["path"]]["assigned_split"], "train")
        self.assertEqual(result["statistics"]["selected_counts"]["test_known"], 2)

    def test_cross_label_duplicate_fails(self):
        data = (self.images / self.rows[0]["path"]).read_bytes()
        (self.images / self.rows[-1]["path"]).write_bytes(data)
        with self.assertRaisesRegex(ValueError, "conflicting labels/sources"):
            self.run_build()
        self.assertFalse((self.root / "out").exists())

    def test_moved_historical_content_cannot_be_relabelled_as_unknown(self):
        previous = self.images / self.rows[0]["path"]
        destination = self.images / "OODTest/TestOOD/moved.png"
        previous.rename(destination)
        with self.assertRaisesRegex(ValueError, "Historical image content carries conflicting"):
            self.run_build()

    def test_invalid_image_fails_before_output(self):
        (self.images / self.rows[0]["path"]).write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "undecodable image"):
            self.run_build()
        self.assertFalse((self.root / "out").exists())

    def test_unexpected_root_and_species_directories_fail(self):
        for path in ("UnknownRoot", "Known/Parent/NewSpecies", "NearDev/Parent/NewUnknown"):
            with self.subTest(path=path):
                directory = self.images / path
                directory.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "Unsupported image directory/class"):
                    self.run_build()
                directory.rmdir()

    def test_unknown_test_and_dev_sources_must_be_disjoint(self):
        self.classes["NearTest"] = {"Parent/DevUnknown": 0}
        self.save_seed()
        with self.assertRaisesRegex(ValueError, "TEST/DEV source overlap"):
            load_seed(self.seed_dir)

    def test_missing_training_support_fails_without_reshuffling(self):
        for row in self.rows:
            if row["split"] == "train":
                (self.images / row["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "no independent training"):
            self.run_build()

    def test_new_unknown_locked_source_stays_test_and_dry_run_does_not_write(self):
        path = self.images / "NearTest/Parent/TestUnknown/brand_new.png"
        path.write_bytes(picture(451))
        result = self.run_build(dry_run=True)
        self.assertEqual(result["statistics"]["selected_counts"]["test_intra"], 3)
        self.assertFalse((self.root / "out").exists())

    def test_taxonomy_tamper_and_image_symlink_rejected(self):
        (self.seed_dir / "tree.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Taxonomy seed hash"):
            self.run_build()
        (self.seed_dir / "tree.npy").write_bytes(self.taxonomy["tree.npy"])
        link = self.images / "Known/Parent/Species/alias.png"
        link.symlink_to(self.images / self.rows[0]["path"])
        with self.assertRaisesRegex(ValueError, "Unsupported file"):
            self.run_build()

    def test_flat_dataset_can_share_parent_with_images_and_remains_immutable(self):
        self.images.rename(self.root / "image")
        result = build(self.root, "image", ".", "seed", dry_run=True)
        self.assertEqual(result["statistics"]["inventory_count"], 14)
        # The seed directory is deliberately outside the managed dataset.
        data = self.root / "data"
        data.mkdir()
        (self.root / "image").rename(data / "image")
        result = build(self.root, "data/image", "data", "seed")
        self.assertTrue((data / "gt_train.txt").is_file())
        self.assertTrue((data / "image").is_dir())
        self.assertEqual(result, build(self.root, "data/image", "data", "seed"))
        original = (data / "gt_train.txt").read_bytes()
        (data / "gt_train.txt").write_bytes(b"tampered\n")
        with self.assertRaisesRegex(ValueError, "Existing dataset differs"):
            build(self.root, "data/image", "data", "seed")
        self.assertNotEqual(original, (data / "gt_train.txt").read_bytes())

    def test_frozen_content_order_survives_filename_sort_reversal(self):
        order = self.freeze_order()
        for index, row in enumerate(self.rows):
            path = self.images / row["path"]
            path.rename(path.with_name("renamed_%04d.png" % (len(self.rows) - index)))
        result = build(self.root, "images", "renamed", "seed")
        new_rows = [json.loads(line) for line in (self.root / "renamed/inventory.jsonl").read_text().splitlines()]
        for split in SPLITS:
            actual = sorted((r for r in new_rows if r["assigned_split"] == split and r["disposition"] == "selected"),
                            key=lambda row: row["manifest_index"])
            self.assertEqual([{"sha256": r["sha256"], "label": r["label"]} for r in actual], order["splits"][split])
        self.assertTrue(result["reconciliation"]["main_experiment_preserved"]["content_labels_and_row_order_unchanged"])

    def test_missing_frozen_identity_fails_instead_of_shrinking_main_test(self):
        self.freeze_order()
        (self.images / self.rows[4]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing frozen main-experiment identity"):
            build(self.root, "images", "missing", "seed")
        self.assertFalse((self.root / "missing").exists())

    def test_restored_historical_test_is_separate_and_inactive(self):
        data = picture(987)
        path = self.add_baseline("NearTest", "test_intra", "restored.png", data)
        (self.images / path).unlink()
        self.save_seed()
        self.freeze_order({"test_intra": [digest(data)], "test_extra": []})
        (self.images / path).write_bytes(data)
        result = build(self.root, "images", "restored", "seed")
        self.assertEqual(result["statistics"]["selected_counts"]["test_intra"], 2)
        self.assertEqual(result["statistics"]["inactive_restored_test_count"], 1)
        self.assertTrue(result["statistics"]["all_images_accounted_for"])
        protocol = json.loads((self.root / "restored/protocol.json").read_text())
        meta = protocol["restored_test_manifests"]["test_intra"]
        self.assertFalse(meta["active"])
        self.assertFalse(meta["gradient_training"])
        self.assertEqual(meta["count"], 1)
        self.assertIn("restored.png", (self.root / meta["path"]).read_text())

    def test_unapproved_addition_cannot_expand_frozen_main_split(self):
        self.freeze_order()
        (self.images / "NearTest/Parent/TestUnknown/new.png").write_bytes(picture(888))
        with self.assertRaisesRegex(ValueError, "Inventory differs from frozen experiment"):
            build(self.root, "images", "expanded", "seed")

    def test_frozen_manifest_order_tamper_is_rejected(self):
        self.freeze_order()
        (self.seed_dir / "manifest_order.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Frozen manifest order hash"):
            build(self.root, "images", "tampered", "seed")


if __name__ == "__main__":
    unittest.main()
