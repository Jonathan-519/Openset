"""Strict D05 import, with Morphology's own frozen-development TEST gate.

The source import contract is intentionally identical to the reviewed Recovery
contract. A Morphology suite never pretends to have Recovery's configuration or
signature in order to open a TEST cache.
"""
from taxosafe_recovery.importer import (
    ARM_ID, FrozenD05, inspect_d05, load_d05, load_parent_cache,
    _directory, _json, _hash, _refresh,
)


def load_parent_test_cache(parent, morphology_suite):
    """Open frozen Discovery TEST features after Morphology DEV is immutable."""
    from taxosafe_discovery import backend, runner as old_runner
    from . import protocol, reporting
    info = _refresh(parent)
    suite = _directory(morphology_suite)
    snapshot = _json(suite / "snapshot.json")
    cfg = protocol.validate_config(_json(suite / "config.json"))
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("source_binding") != info["binding"]
            or _json(suite / "source_binding.json") != info["binding"]
            or snapshot.get("signature") != protocol.signature(cfg, info["binding"])):
        raise ValueError("Morphology TEST cache request belongs to another D05 parent or protocol")
    old_runner._regular(suite / "dev_selection.json")
    reporting.freeze_dev_selection(suite)
    old_runner._verify_stage(info["directory"], None, "cache_test", info["snapshot"])
    cache, receipt = backend._load_cache(info["directory"], "test", info["config"], info["reference"])
    if receipt.get("dev_selection_sha256") != _hash(info["directory"] / "dev_selection.json"):
        raise ValueError("Original TEST cache is not bound to its original DEV decision")
    if receipt["inference_spec_sha256"] != info["training"]["inference_spec_sha256"]:
        raise ValueError("Original TEST cache used a different frozen representation")
    return cache, receipt
