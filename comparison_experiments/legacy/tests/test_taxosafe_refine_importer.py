"""Reviewed code migration is explicit, finite, and independent of old checks."""
import copy
import unittest

from taxosafe_refine.importer import APPROVED_SOURCES, verify_source_signature
from taxosafe_support.protocol import require_signature


def approved_fixture_signature(config_digest="0" * 64, historic=True):
    code, support = list(APPROVED_SOURCES)[0 if historic else 1]
    return {"method": "support_reference_v3", "config": config_digest, "hierarchy": "1" * 64,
            "known_preparation_audit": "2" * 64, "code": code, "support_code": support}


class ReferenceImportSignatureTest(unittest.TestCase):
    def test_only_reviewed_migration_allowed_and_original_check_still_strict(self):
        source, runtime = approved_fixture_signature(), approved_fixture_signature(historic=False)
        with self.assertRaises(ValueError):
            require_signature(source, runtime)
        migration = verify_source_signature(source, runtime)
        self.assertEqual(migration["source_commit"], "54dff0ed81771e4cbc14b844f22ebea06e3f9c8d")
        self.assertEqual(migration["runtime_commit"], "4394be543badc2a7def2fa5a60b45b531b5a861d")
        self.assertEqual(migration["source_signature"], source)
        self.assertEqual(migration["runtime_signature"], runtime)

    def test_unapproved_source_or_runtime_and_noncode_changes_fail_closed(self):
        original = approved_fixture_signature()
        for key in original:
            changed = copy.deepcopy(original)
            changed[key] = "9" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_source_signature(original, changed)
            with self.subTest(source_key=key), self.assertRaises(ValueError):
                verify_source_signature(changed, original)
        added = dict(original, unidentified_provenance="a" * 64)
        with self.assertRaises(ValueError):
            verify_source_signature(original, added)
        missing = dict(original)
        missing.pop("known_preparation_audit")
        with self.assertRaises(ValueError):
            verify_source_signature(missing, original)


if __name__ == "__main__":
    unittest.main()
