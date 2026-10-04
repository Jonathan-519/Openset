"""Strict reject-only decoder for the empirical-frontier experiment."""
import copy

from taxosafe_parentrisk import decoder as parentrisk

SCHEMA_VERSION = "frontier_v1"
DECODER = "frontier"
digest = parentrisk.digest


def make_router(baseline, meta, rules=None):
    state = parentrisk.make_router(baseline, meta, mode="audit", reject_rules=rules)
    state.update(schema_version=SCHEMA_VERSION, decoder=DECODER)
    return state


def validate_router(router, meta):
    if (not isinstance(router, dict) or router.get("schema_version") != SCHEMA_VERSION
            or router.get("decoder") != DECODER or router.get("mode") != "audit"
            or router.get("meta") != meta or router.get("parent_rule", "missing") is not None):
        raise ValueError("Frontier schema, decoder, reject-only mode or hierarchy mismatch")
    # Conversion is local and explicit; it cannot change the archived object.
    converted = copy.copy(router)
    converted.update(schema_version=parentrisk.SCHEMA_VERSION, decoder=parentrisk.DECODER)
    parentrisk.validate_router(converted, meta)
    return converted


def apply_router(records, router, meta):
    converted = validate_router(router, meta)
    rows = parentrisk.apply_router(records, converted, meta)
    for row in rows:
        row.update(decoder=DECODER,
                   root_score_type="frontier_route_indicator_not_probability",
                   local_score_type="frontier_route_indicator_not_probability")
    return rows
