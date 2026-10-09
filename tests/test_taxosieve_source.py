"""Exercise reviewed signature migration against the actual current core hash.

These checks do not grant compatibility by mocking signatures or changing the
allowlist: the native signature is computed from the shipped reference config,
taxonomy, preparation audit, and runtime files. Historical sources retain every
non-code field, and only their two reviewed code pairs may migrate.
"""
import copy
from pathlib import Path
import unittest

from taxosieve import source
from taxosafe_support import protocol


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PAIRS = {
    ("7b6b7e02dc7b0181d93467e6d5c25bc1e42823d3d3c2c90088c28e8aa66b671b",
     "ef5526b8390b838473272eaaf56dcdb128e37e9e2c3a543896d8cd5a30e9aed4"):
        "54dff0ed81771e4cbc14b844f22ebea06e3f9c8d",
    ("7b6b7e02dc7b0181d93467e6d5c25bc1e42823d3d3c2c90088c28e8aa66b671b",
     "77e239bed3c7a8c89d1fbcfaa8da0d097c202a26a68013dbeb644dba8f22faae"):
        "4394be543badc2a7def2fa5a60b45b531b5a861d",
}


class SourceSignatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = protocol.effective_config(ROOT / "configs/taxosieve_reference.yml", "main", 1)
        cls.native = protocol.signature(cls.config)
        cls.native_pair = (cls.native["code"], cls.native["support_code"])

    def _signature(self, pair):
        signature = copy.deepcopy(self.native)
        signature["code"], signature["support_code"] = pair
        return signature

    def test_actual_native_signature_is_reviewed_and_matches_itself(self):
        self.assertIn(self.native_pair, source.NATIVE_SOURCE_DIGESTS,
            "Runtime changed: re-establish equivalence before reviewing new code hashes")
        self.assertEqual(set(self.native), {
            "config", "hierarchy", "known_preparation_audit", "code", "support_code", "method"})
        result = source.verify_source_signature(self.native, copy.deepcopy(self.native))
        self.assertEqual(result["source_signature"], self.native)
        self.assertEqual(result["runtime_signature"], self.native)
        self.assertEqual(result["source_commit"], source.NATIVE_SOURCE_DIGESTS[self.native_pair])
        self.assertEqual(result["source_commit"], result["runtime_commit"])

    def test_both_reviewed_historical_sources_migrate_without_mutating_provenance(self):
        for pair, commit in HISTORICAL_PAIRS.items():
            with self.subTest(source_commit=commit):
                historical = self._signature(pair)
                before = copy.deepcopy(historical)
                result = source.verify_source_signature(historical, self.native)
                self.assertEqual(result["source_commit"], commit)
                self.assertEqual(result["runtime_commit"], source.NATIVE_SOURCE_DIGESTS[self.native_pair])
                self.assertEqual(result["source_signature"], before)
                self.assertEqual(result["runtime_signature"], self.native)
                self.assertEqual(historical, before)
                # Migration is explicit at the importer; native stage equality
                # remains strict and cannot silently accept historical hashes.
                with self.assertRaises(ValueError):
                    protocol.require_signature(self.native, historical)

    def test_approved_pairs_can_only_exchange_code_fields(self):
        pairs = [self.native_pair, *HISTORICAL_PAIRS]
        for source_pair in pairs:
            for runtime_pair in pairs:
                with self.subTest(source=source_pair, runtime=runtime_pair):
                    previous, current = self._signature(source_pair), self._signature(runtime_pair)
                    result = source.verify_source_signature(previous, current)
                    self.assertEqual(result["source_signature"], previous)
                    self.assertEqual(result["runtime_signature"], current)

    def test_config_taxonomy_or_audit_changes_are_rejected(self):
        for pair in [self.native_pair, *HISTORICAL_PAIRS]:
            for side in ("source", "runtime"):
                for field in ("config", "hierarchy", "known_preparation_audit"):
                    with self.subTest(pair=pair, side=side, field=field):
                        previous, current = self._signature(pair), copy.deepcopy(self.native)
                        changed = previous if side == "source" else current
                        changed[field] = "0" * 64
                        self.assertNotEqual(changed[field], self.native[field])
                        with self.assertRaisesRegex(ValueError, "configuration, taxonomy or preparation audit"):
                            source.verify_source_signature(previous, current)

    def test_unknown_or_mixed_code_pairs_are_rejected_for_both_sides(self):
        historical_code, historical_support = next(iter(HISTORICAL_PAIRS))
        unreviewed = [("0" * 64, self.native_pair[1]),
                      (self.native_pair[0], "0" * 64),
                      ("0" * 64, "0" * 64),
                      (self.native_pair[0], historical_support),
                      (historical_code, self.native_pair[1])]
        for pair in unreviewed:
            self.assertNotIn(pair, source.APPROVED_SOURCES)
            for side in ("source", "runtime"):
                with self.subTest(pair=pair, side=side):
                    previous, current = copy.deepcopy(self.native), copy.deepcopy(self.native)
                    changed = previous if side == "source" else current
                    changed["code"], changed["support_code"] = pair
                    with self.assertRaisesRegex(ValueError, "reviewed digest allowlist"):
                        source.verify_source_signature(previous, current)
            # Equal unknown source/runtime hashes do not bypass approval.
            unknown = self._signature(pair)
            with self.assertRaisesRegex(ValueError, "reviewed digest allowlist"):
                source.verify_source_signature(unknown, copy.deepcopy(unknown))

    def test_missing_or_added_fields_and_non_reference_methods_are_rejected(self):
        for pair in [self.native_pair, *HISTORICAL_PAIRS]:
            for side in ("source", "runtime"):
                for change in ("missing_audit", "extra_field", "other_method"):
                    with self.subTest(pair=pair, side=side, change=change):
                        previous, current = self._signature(pair), copy.deepcopy(self.native)
                        changed = previous if side == "source" else current
                        if change == "missing_audit":
                            del changed["known_preparation_audit"]
                        elif change == "extra_field":
                            changed["unreviewed_field"] = "value"
                        else:
                            changed["method"] = "support_relation_v1"
                        with self.assertRaises(ValueError):
                            source.verify_source_signature(previous, current)
        non_reference = copy.deepcopy(self.native)
        non_reference["method"] = "support_relation_v1"
        with self.assertRaisesRegex(ValueError, "Only reference-v3"):
            source.verify_source_signature(non_reference, copy.deepcopy(non_reference))


if __name__ == "__main__":
    unittest.main()
