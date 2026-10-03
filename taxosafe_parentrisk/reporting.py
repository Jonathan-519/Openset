"""Paired harm and coverage audits with explicit finite-sample limitations."""
from collections import defaultdict

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_geometry import calibration as geometry
from .folds import VALIDATION_SCOPE, source_key, unique_records


def _rate(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "rate": None if not denominator else float(numerator / denominator)}


def _correct(row):
    if row["status"] == "known":
        return (row["prediction_type"] == "known" and row["leaf"] == row["true_leaf"]
                and row["parent"] == row["true_parent"])
    if row["status"] == "intra":
        return row["prediction_type"] == "intra_unknown" and row["parent"] == row["true_parent"]
    return row["prediction_type"] == "global_unknown"


def _paired(before, after):
    old = {base._digest(r): r for r in unique_records(before)}
    new = {base._digest(r): r for r in unique_records(after)}
    if old.keys() != new.keys():
        raise ValueError("Paired audit requires the exact same unique content hash set")
    keys = sorted(old)
    for digest in keys:
        if any(old[digest].get(k) != new[digest].get(k) for k in
               ("status", "split", "source", "species", "true_parent", "true_leaf")):
            raise ValueError("Paired audit annotations changed: " + digest)
    return [old[k] for k in keys], [new[k] for k in keys]


def _parent_path(row):
    if row["status"] != "intra" or row["candidate_parent"] != row["true_parent"]:
        return False
    threshold = row.get("baseline_parent_threshold", row.get("parent_threshold"))
    score = row.get("parent_membership_score")
    passed = (score is not None and threshold is not None and
              np.isfinite([float(score), float(threshold)]).all() and float(score) >= float(threshold))
    return bool(passed or row["prediction_type"] != "global_unknown")


def _recall(rows, meta):
    result = {}
    parent_count = len(meta["parent_names"])
    for status in ("known", "intra"):
        selected = [r for r in rows if r["status"] == status]
        result[status] = {}
        for namespace, field, name in (("support_evidence", "parent_logits", "support"),
                                      ("encoder_evidence", "parent_text_logits", "text")):
            available = [r for r in selected if field in r.get(namespace, {})]
            top1, top2 = 0, 0
            for row in available:
                scores = np.asarray(row[namespace][field], dtype=float)
                if scores.shape != (parent_count,) or not np.isfinite(scores).all():
                    raise ValueError("Invalid full parent vector for recall: " + name)
                ranking = np.argsort(-scores, kind="stable")
                top1 += int(ranking[0] == row["true_parent"])
                top2 += int(row["true_parent"] in ranking[:2])
            result[status][name] = {"top1": _rate(top1, len(available)),
                                    "top2": _rate(top2, len(available)),
                                    "missing_count": len(selected) - len(available)}
    return result


def paired_report(before, after, meta):
    """Audit paired outputs; truth is used here, never by the routing function."""
    geometry._validate_meta(meta)
    old, new = _paired(before, after)
    baseline, selected = base.evaluate_records(old, meta), base.evaluate_records(new, meta)
    known = [i for i, row in enumerate(old) if row["status"] == "known"]
    near = [i for i, row in enumerate(old) if row["status"] == "intra"]
    known_harm = [i for i in known if _correct(old[i]) and not _correct(new[i])]
    near_harm = [i for i in near if _correct(old[i]) and not _correct(new[i])]
    paths = [i for i in near if _parent_path(old[i])]
    lost_paths = [i for i in paths if new[i]["prediction_type"] == "global_unknown"]
    retained_paths = [i for i in paths if new[i]["prediction_type"] != "global_unknown"
                      and new[i]["parent"] == old[i]["true_parent"]]
    risks = {"known_added_harm": _rate(len(known_harm), len(known)),
             "near_added_harm": _rate(len(near_harm), len(near)),
             "near_parent_path_loss": _rate(len(lost_paths), len(near)),
             "near_parent_path_retention": _rate(len(retained_paths), len(paths)),
             "near_root_rejection": _rate(sum(new[i]["prediction_type"] == "global_unknown" for i in near), len(near))}
    risks["known_added_harm"]["image_sha256"] = [base._digest(old[i]) for i in known_harm]
    risks["near_added_harm"]["image_sha256"] = [base._digest(old[i]) for i in near_harm]
    risks["near_parent_path_loss"]["image_sha256"] = [base._digest(old[i]) for i in lost_paths]
    leaves = {}
    for leaf, name in enumerate(meta["leaf_names"]):
        indices = [i for i in known if old[i]["true_leaf"] == leaf]
        leaves[name] = {"leaf_id": leaf, "sample_count": len(indices),
                        "baseline_correct": sum(_correct(old[i]) for i in indices),
                        "selected_correct": sum(_correct(new[i]) for i in indices),
                        "added_harm": sum(i in known_harm for i in indices),
                        "evidence_status": "not_evaluable" if not indices else
                                           "insufficient_evidence" if len(indices) < 5 else "observed"}
    sources = {s: {} for s in ("intra", "extra")}
    macro = {}
    for status in sources:
        grouped = defaultdict(list)
        for i, row in enumerate(old):
            if row["status"] == status:
                grouped[source_key(row)].append(i)
        for source, indices in sorted(grouped.items()):
            sources[status][source] = {"sample_count": len(indices),
                "source_names": sorted({str(old[i].get("source") or "unspecified") for i in indices}),
                "baseline_correct": sum(_correct(old[i]) for i in indices),
                "selected_correct": sum(_correct(new[i]) for i in indices),
                "baseline_correct_rate": sum(_correct(old[i]) for i in indices) / len(indices),
                "selected_correct_rate": sum(_correct(new[i]) for i in indices) / len(indices)}
        values = list(sources[status].values())
        macro[status] = {"source_count": len(values),
                         "baseline": None if not values else float(np.mean([v["baseline_correct_rate"] for v in values])),
                         "selected": None if not values else float(np.mean([v["selected_correct_rate"] for v in values]))}
    bc, nc = baseline["counts"], selected["counts"]
    precision_preserved = ((nc["leaf_outputs"] == 0) if bc["leaf_outputs"] == 0 else
                           nc["leaf_outputs"] > 0 and
                           nc["known_correct"] * bc["leaf_outputs"] >= bc["known_correct"] * nc["leaf_outputs"])
    audit = {"known_correct_images_preserved": not known_harm,
             "near_correct_images_preserved": not near_harm,
             "unknown_source_correct_counts_preserved": all(v["selected_correct"] >= v["baseline_correct"]
                                                              for group in sources.values() for v in group.values()),
             "leaf_precision_preserved": bool(precision_preserved),
             "near_parent_paths_preserved": not lost_paths}
    audit["passed"] = all(audit.values())
    return {"schema_version": "parentrisk_paired_report_v1", "unit": "unique_image_content_sha256",
            "validation_scope": VALIDATION_SCOPE, "independent_model_level_validation": False,
            "unique_image_count": len(old), "baseline": baseline, "selected": selected,
            "risks": risks, **risks, "per_known_leaf": leaves, "per_source": sources,
            "source_macro": macro, "parent_recall": _recall(old, meta), "preservation_audit": audit,
            "evidence_note": "Observed paired protection is conditional on the frozen reference; it is not a population guarantee. Empty and rare classes are explicitly unevaluable or insufficient."}


def score_diagnostics(records):
    """Continuous component ranking, with fixed raw support top-1 alignment."""
    rows = unique_records(records)
    status = np.asarray([r["status"] for r in rows], dtype=str)
    columns = {key: [r[key] for r in rows] for key in geometry.EVIDENCE_FIELDS}
    for key in ("text_z", "membership_z", "geometry_z"):
        if rows and all(key in r.get("parent_evidence", {}) for r in rows):
            columns["selected_parent_" + key] = [r["parent_evidence"][key][int(np.argmax(r["support_evidence"]["parent_logits"]))]
                                                for r in rows]
    components = {}
    for key, column in columns.items():
        values = np.asarray(column, dtype=float)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError("Continuous diagnostics require finite scalar scores: " + key)
        parent = "parent" in key
        pos = values[status != "extra"] if parent else values[status == "known"]
        neg = np.sort(values[status == "extra"] if parent else values[status == "intra"])
        auc = None
        if len(pos) and len(neg):
            left, right = np.searchsorted(neg, pos, side="left"), np.searchsorted(neg, pos, side="right")
            auc = float((left + right).sum() / (2. * len(pos) * len(neg)))
        components[key] = {"auroc": auc, "positive_count": len(pos), "negative_count": len(neg),
                           "positive_statuses": ["known", "intra"] if parent else ["known"],
                           "negative_statuses": ["extra"] if parent else ["intra"]}
    return {"unique_images": len(rows), "components": components,
            "alignment": "raw_support_top1_parent_before_gating",
            "score_note": "Continuous evidence AUROC does not imply final route improvement; route indicators are excluded."}


def rule_support(rows, before, after, meta, rule_indices=None):
    """Summarize each rule's own hits instead of borrowing action-wide gains.

    When rules overlap, these are observed matched-region changes, not causal
    attribution. Per-rule ablations are needed to infer isolated effects.
    """
    old, new = _paired(before, after)
    if {base._digest(r) for r in unique_records(rows)} != {base._digest(r) for r in old}:
        raise ValueError("Rule support rows must match paired prediction hashes")
    indices = sorted(set(rule_indices if rule_indices is not None else
                         [index for r in new for index in r.get("applied_rule_indices", [])]))
    report = {}
    for index in indices:
        matches = [i for i, row in enumerate(new) if index in row.get("applied_rule_indices", [])]
        local = paired_report([old[i] for i in matches], [new[i] for i in matches], meta)
        gains = [{"status": status, "source": source, **value}
                 for status, sources in local["per_source"].items() for source, value in sources.items()
                 if value["selected_correct"] > value["baseline_correct"]]
        regions = {}
        for parent, name in enumerate(meta["parent_names"]):
            region = [i for i in matches if new[i].get("route_parent", new[i]["candidate_parent"]) == parent]
            regional = paired_report([old[i] for i in region], [new[i] for i in region], meta)
            regional_gains = [{"status": status, "source": source, **value}
                              for status, sources in regional["per_source"].items() for source, value in sources.items()
                              if value["selected_correct"] > value["baseline_correct"]]
            regions[name] = {"parent_id": parent, "matched_count": len(region),
                             "improved_source_count": len(regional_gains), "improved_sources": regional_gains,
                             "per_source": regional["per_source"], "per_known_leaf": regional["per_known_leaf"],
                             "risks": regional["risks"],
                             "evidence_status": "not_evaluable" if not region else
                                                "insufficient_evidence" if len(regional_gains) < 2 else "observed"}
        report[str(index)] = {"matched_count": len(matches), "improved_source_count": len(gains),
                              "improved_sources": gains, "per_known_leaf": local["per_known_leaf"],
                              "per_source": local["per_source"], "risks": local["risks"],
                              "per_parent": regions,
                              "evidence_status": "not_evaluable" if not matches else
                                                 "insufficient_evidence" if len(gains) < 2 else "observed"}
    return {"rules": report, "attribution_note": "Matched regions may overlap; changes are not causal per-rule attribution."}
