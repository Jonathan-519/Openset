"""Content-grouped DEV folds for a frozen-reference postprocessor audit.

Splitting uses annotations only for stratification. No score, prediction, or
operating point influences assignment. This is deliberately not advertised as
independent validation of the already selected reference model.
"""
from collections import defaultdict
import hashlib
import json

import numpy as np

from taxosafe_geometry import calibration as geometry
from taxosafe_support import calibration as base
from taxosafe_dcbs.protocol import normalized_name


VALIDATION_SCOPE = "postprocessor_conditional_on_frozen_reference"


def source_key(row):
    """Match the project's source identity rules while retaining display names."""
    return normalized_name(str(row.get("source") or "unspecified")) or "unspecified"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def unique_records(records):
    """Deduplicate only after checking every collected evidence namespace."""
    records = list(records)
    seen = {}
    for row in records:
        digest = base._digest(row)
        evidence = {}
        for name in ("parent_evidence", "encoder_evidence"):
            if name in row:
                try:
                    evidence[name] = _hash(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid finite JSON " + name + ": " + digest) from exc
        if digest in seen and seen[digest] != evidence:
            raise ValueError("Same content has inconsistent parent/encoder evidence: " + digest)
        seen[digest] = evidence
    return geometry.unique_records(records)


def build_folds(records, meta, n_splits=4, seed=0):
    """Assign each unique image to exactly one held fold, including known.

    Entire unknown sources are held together within each status. Known images
    are balanced by leaf, then parent, then total count; singleton leaves stay
    in the plan and are marked as insufficient evidence in the report.
    """
    geometry._validate_meta(meta)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or not 2 <= n_splits <= 32:
        raise ValueError("n_splits must be an integer in [2,32]")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    n_splits, seed = int(n_splits), int(seed)
    original = list(records)
    rows = sorted(unique_records(original), key=base._digest)
    if not rows:
        raise ValueError("DEV fold planning requires nonempty records")
    _, _, mapping = base._hierarchy(meta)
    for row in rows:
        status = row.get("status")
        if status not in base.STATUSES or row.get("split") != "val_" + status:
            raise ValueError("Fold planning requires DEV records only; test fitting is prohibited")
        if status in ("known", "intra"):
            parent = row.get("true_parent")
            if isinstance(parent, bool) or not isinstance(parent, (int, np.integer)) or not 0 <= parent < len(meta["parent_names"]):
                raise ValueError("Invalid DEV true_parent")
        if status == "known":
            leaf = row.get("true_leaf")
            if (isinstance(leaf, bool) or not isinstance(leaf, (int, np.integer)) or
                    not 0 <= leaf < len(mapping) or mapping[leaf] != row["true_parent"]):
                raise ValueError("Invalid DEV known leaf/parent annotation")

    def order(*parts):
        return _hash([seed, *parts])

    held = [[] for _ in range(n_splits)]
    known = defaultdict(list)
    for row in rows:
        if row["status"] == "known":
            known[int(row["true_leaf"])].append(row)
    leaf_counts = [defaultdict(int) for _ in held]
    parent_counts = [defaultdict(int) for _ in held]
    known_counts = [0] * n_splits
    for leaf in sorted(known, key=lambda c: (-len(known[c]), order("leaf", c))):
        parent = int(mapping[leaf])
        for row in sorted(known[leaf], key=lambda r: order("image", base._digest(r))):
            index = min(range(n_splits), key=lambda i: (
                leaf_counts[i][leaf], parent_counts[i][parent], known_counts[i],
                order("tie", leaf, base._digest(row), i)))
            held[index].append(base._digest(row))
            leaf_counts[index][leaf] += 1
            parent_counts[index][parent] += 1
            known_counts[index] += 1
    for status in ("intra", "extra"):
        sources = defaultdict(list)
        for row in rows:
            if row["status"] == status:
                sources[source_key(row)].append(base._digest(row))
        for index, source in enumerate(sorted(sources, key=lambda s: order(status, s))):
            held[index % n_splits].extend(sources[source])

    by_hash = {base._digest(row): row for row in rows}
    all_hashes = set(by_hash)
    folds = []
    for index, digest_list in enumerate(held):
        held_hashes = sorted(digest_list)
        fit_hashes = sorted(all_hashes.difference(held_hashes))
        fit_counts = {s: sum(by_hash[h]["status"] == s for h in fit_hashes) for s in base.STATUSES}
        held_counts = {s: sum(by_hash[h]["status"] == s for h in held_hashes) for s in base.STATUSES}
        missing = [s for s in base.STATUSES if fit_counts[s] == 0]
        reason = ("missing_fit_statuses:" + ",".join(missing)) if missing else (None if held_hashes else "empty_held_fold")
        folds.append({"fold_id": index, "fit_image_sha256": fit_hashes,
                      "held_image_sha256": held_hashes,
                      "held_sources": {s: sorted({source_key(by_hash[h])
                                                  for h in held_hashes if by_hash[h]["status"] == s})
                                       for s in ("intra", "extra")},
                      "held_source_names": {s: sorted({str(by_hash[h].get("source") or "unspecified")
                                                       for h in held_hashes if by_hash[h]["status"] == s})
                                            for s in ("intra", "extra")},
                      "fit_counts": fit_counts, "held_counts": held_counts,
                      "known_fit_n": fit_counts["known"], "known_held_n": held_counts["known"],
                      "known_held_evidence": "not_evaluable" if not held_counts["known"] else
                                             "insufficient_evidence" if held_counts["known"] < 5 else "observed",
                      "usable": reason is None, "reason": reason})
    assigned = [h for fold in folds for h in fold["held_image_sha256"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != all_hashes:
        raise AssertionError("Every unique content hash must be held exactly once")
    plan = {"schema_version": "parentrisk_outer_folds_v1", "n_splits": n_splits, "seed": seed,
            "validation_scope": VALIDATION_SCOPE, "independent_model_level_validation": False,
            "unit": "unique_image_content_sha256", "unique_image_count": len(rows),
            "duplicate_record_count": len(original) - len(rows),
            "assignment_rule": "known_leaf_parent_balance;unknown_whole_source_round_robin;score_independent",
            "folds": folds}
    # Duplicate aliases do not affect the fold identity or its binding digest.
    plan["sha256"] = _hash({k: v for k, v in plan.items() if k != "duplicate_record_count"})
    return plan
