"""Pure calibration helpers shared by TaxoSieve and its fixed D05 evidence model.

Hashing deliberately uses the original compact JSON encoding. It differs from
support artifact receipt hashing and must not be replaced by io.object_hash.
The audit/fold algorithms retain their original serialization and ordering.
"""
import hashlib
import json
from collections import defaultdict

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support.io import normalized_name

CONTEXT = dict(validation_scope="exploratory_calibration_conditional_on_previously_selected_model",
               independent_model_level_validation=False, confirmatory_validation=False,
               test_used_for_fitting=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def _correct(row):
    if row["status"] == "known":
        return row["prediction_type"] == "known" and row["leaf"] == row["true_leaf"] and row["parent"] == row["true_parent"]
    return (row["prediction_type"] == "intra_unknown" and row["parent"] == row["true_parent"]) if row["status"] == "intra" else row["prediction_type"] == "global_unknown"

def _wrong_leaf(row):
    return row["prediction_type"] == "known" and not _correct(row)

def paired_audit(before, after, meta):
    old = {base._digest(r): r for r in base.unique_records(before)}
    new = {base._digest(r): r for r in base.unique_records(after)}
    if set(old) != set(new):
        raise ValueError("Paired audit hashes differ")
    groups = defaultdict(list)
    known_lost, known_gained, wrong_added, near_lost, extra_lost, path_lost = [], [], [], [], [], []
    for key in sorted(old):
        b, a = old[key], new[key]
        if any(b.get(k) != a.get(k) for k in ("status", "split", "true_leaf", "true_parent", "source", "species")):
            raise ValueError("Paired audit annotations differ")
        status = b["status"]
        if status == "known":
            if _correct(b) and not _correct(a): known_lost.append(key)
            if not _correct(b) and _correct(a): known_gained.append(key)
        elif status == "intra":
            if _correct(b) and not _correct(a): near_lost.append(key)
            if b["prediction_type"] != "global_unknown" and b["parent"] == b["true_parent"] and (a["prediction_type"] == "global_unknown" or a["parent"] != b["true_parent"]): path_lost.append(key)
        elif _correct(b) and not _correct(a): extra_lost.append(key)
        if not _wrong_leaf(b) and _wrong_leaf(a): wrong_added.append(key)
        name = str(b.get("source") or "unspecified")
        group = int(b["true_leaf"]) if status == "known" else normalized_name(name)
        groups[(status, group)].append((b, a))
    per_source, per_leaf = {}, {}
    for (status, group), pairs in groups.items():
        entry = dict(sample_count=len(pairs), reference_correct=sum(_correct(b) for b, a in pairs),
                     selected_correct=sum(_correct(a) for b, a in pairs),
                     reference_false_leaves=sum(_wrong_leaf(b) for b, a in pairs),
                     selected_false_leaves=sum(_wrong_leaf(a) for b, a in pairs),
                     lost_correct=sum(_correct(b) and not _correct(a) for b, a in pairs),
                     gained_correct=sum(not _correct(b) and _correct(a) for b, a in pairs),
                     evidence_status="insufficient_evidence" if len(pairs) < 5 else "observed")
        if status == "known":
            per_leaf[meta["leaf_names"][group]] = entry
        else:
            entry.update(status=status, source=group, source_names=sorted({str(b.get("source") or "unspecified") for b, a in pairs}))
            per_source[status + ":" + group] = entry
    for name in meta["leaf_names"]:
        per_leaf.setdefault(name, dict(sample_count=0, reference_correct=0, selected_correct=0,
                         reference_false_leaves=0, selected_false_leaves=0, lost_correct=0, gained_correct=0,
                         evidence_status="not_evaluable"))
    macro = {}
    for status in ("intra", "extra"):
        values = [v for v in per_source.values() if v["status"] == status]
        macro[status] = {"source_count": len(values), **{key: None if not values else float(np.mean([v[key + "_correct"] / v["sample_count"] for v in values])) for key in ("reference", "selected")}}
    safe_sources = all(v["selected_correct"] >= v["reference_correct"] and v["selected_false_leaves"] <= v["reference_false_leaves"] for v in per_source.values())
    gains = [key for key, v in per_source.items() if v["selected_correct"] > v["reference_correct"] or v["selected_false_leaves"] < v["reference_false_leaves"]]
    return dict(**CONTEXT, reference=base.evaluate_records(list(old.values()), meta), selected=base.evaluate_records(list(new.values()), meta),
                known_lost_correct=len(known_lost), known_lost_correct_sha256=known_lost,
                known_gained_correct=len(known_gained), known_gained_correct_sha256=known_gained,
                new_wrong_leaf_count=len(wrong_added), new_wrong_leaf_sha256=wrong_added,
                known_new_wrong_leaf_count=sum(old[k]["status"] == "known" for k in wrong_added),
                near_lost_correct=len(near_lost), extra_lost_correct=len(extra_lost), near_parent_path_loss=len(path_lost),
                per_known_leaf=per_leaf, per_unknown_source=per_source, source_macro=macro,
                benefiting_sources=gains, unknown_sources_preserved=bool(safe_sources),
                passed=not known_lost and bool(safe_sources) and bool(gains))

def _folds(rows, settings):
    known_held = [[] for _ in range(3)]
    leaves = defaultdict(list)
    for row in rows:
        if row["status"] == "known": leaves[row["true_leaf"]].append(row)
    for leaf in sorted(leaves, key=lambda key: (-len(leaves[key]), key)):
        counts = [0] * len(known_held)
        for row in sorted(leaves[leaf], key=lambda r: _hash([settings["seed"], base._digest(r)])):
            fold = min(range(len(known_held)), key=lambda index: (counts[index], len(known_held[index]), index))
            known_held[fold].append(base._digest(row)); counts[fold] += 1
    result = [dict(fold_id="known_"+str(index), kind="known", held_image_sha256=sorted(values)) for index, values in enumerate(known_held)]
    sources = defaultdict(list)
    for row in rows:
        if row["status"] != "known": sources[(row["status"], normalized_name(str(row.get("source") or "unspecified")))].append(base._digest(row))
    result.extend(dict(fold_id="source_"+str(index), kind="unknown_source", status=status, source=source,
                       held_image_sha256=sorted(values)) for index, ((status, source), values) in enumerate(sorted(sources.items())))
    hashes = {base._digest(row) for row in rows}
    for fold in result:
        fold["fit_image_sha256"] = sorted(hashes - set(fold["held_image_sha256"]))
    return result


def _masks(counts, totals):
    k, n, e, l = counts.T
    known = 10*k > 9*totals[0]
    precision = (l > 0) & (10*k > 9*l)
    four = known & precision & (20*n >= 17*totals[1]) & (10*e > 9*totals[2])
    return four, known & precision, known
