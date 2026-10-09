"""Read-only failure audit of saved predictions; DEV only unless requested.

No model, calibration, threshold search, stage receipt, or router is created.
The optional JSON destination must be a new file outside the source run.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosafe_support import calibration as base
from taxosafe_dcbs.protocol import normalized_name


def _ratio(numerator, denominator):
    return {"numerator": int(numerator), "denominator": int(denominator),
            "rate": None if not denominator else numerator / denominator}


def _correct(row):
    if row["status"] == "known":
        return (row["prediction_type"] == "known" and row["leaf"] == row["true_leaf"]
                and row["parent"] == row["true_parent"])
    if row["status"] == "intra":
        return row["prediction_type"] == "intra_unknown" and row["parent"] == row["true_parent"]
    return row["prediction_type"] == "global_unknown"


def _unique(rows, meta, evaluate_test=False):
    rows = list(rows)
    p, c, mapping = base._hierarchy(meta)
    unique = base.unique_records(rows)
    seen = {}
    # Older deduplication predates these fields. Contradictory aliases must not
    # silently make a new decoder's rule attribution depend on record order.
    fields = ("route_parent", "output_node", "applied_rule_indices",
              "encoder_evidence", "parent_evidence")
    for row in rows:
        digest = base._digest(row)
        if digest in seen and any(seen[digest].get(k) != row.get(k) for k in fields):
            raise ValueError("Conflicting diagnostic evidence for content alias: " + digest)
        seen[digest] = row
    for row in unique:
        status, split = row.get("status"), row.get("split", "")
        expected = ("test_" if evaluate_test else "val_") + str(status)
        if status not in base.STATUSES or split != expected:
            raise ValueError("Expected only " + ("TEST" if evaluate_test else "DEV") + " records: " + str(split))
        kind, parent, leaf = row.get("prediction_type"), row.get("parent"), row.get("leaf")
        if kind not in base.KINDS:
            raise ValueError("Saved terminal predictions are required")
        if kind == "global_unknown":
            valid = parent is None and leaf is None
            node = 0
        elif kind == "intra_unknown":
            valid = type(parent) is int and 0 <= parent < p and leaf is None
            node = None if not valid else 1 + parent
        else:
            valid = (type(parent) is int and 0 <= parent < p and type(leaf) is int
                     and 0 <= leaf < c and int(mapping[leaf]) == parent)
            node = None if not valid else 1 + p + leaf
        if not valid or row.get("output_node", node) != node:
            raise ValueError("Saved prediction is inconsistent with the taxonomy")
        if status in ("known", "intra"):
            truth = row.get("true_parent")
            if type(truth) is not int or not 0 <= truth < p:
                raise ValueError("Known/near truth requires a valid parent")
        if status == "known":
            truth = row.get("true_leaf")
            if type(truth) is not int or not 0 <= truth < c or mapping[truth] != row["true_parent"]:
                raise ValueError("Known truth disagrees with taxonomy")
        if type(row.get("candidate_parent")) is not int or not 0 <= row["candidate_parent"] < p:
            raise ValueError("Saved support candidate_parent is required")
    return unique


def _near_report(rows):
    near = [r for r in rows if r["status"] == "intra"]
    terminal = dict.fromkeys(("correct_parent_fallback", "root_rejection",
                             "leaf_false_acceptance", "wrong_parent_fallback"), 0)
    candidate_decomposition = dict.fromkeys((
        "candidate_wrong_terminal_error", "candidate_wrong_repaired_success",
        "candidate_correct_root", "candidate_correct_leaf",
        "candidate_correct_wrong_parent_fallback", "candidate_correct_parent_fallback"), 0)
    candidate_correct = accepted_correct_route = top2_correct = top2_available = 0
    for row in near:
        kind = row["prediction_type"]
        key = ("correct_parent_fallback" if _correct(row) else "root_rejection"
               if kind == "global_unknown" else "leaf_false_acceptance"
               if kind == "known" else "wrong_parent_fallback")
        terminal[key] += 1
        good_candidate = row["candidate_parent"] == row["true_parent"]
        candidate_correct += int(good_candidate)
        if not good_candidate:
            key = "candidate_wrong_repaired_success" if _correct(row) else "candidate_wrong_terminal_error"
        else:
            key = ("candidate_correct_root" if kind == "global_unknown" else
                   "candidate_correct_leaf" if kind == "known" else
                   "candidate_correct_parent_fallback" if _correct(row) else
                   "candidate_correct_wrong_parent_fallback")
        candidate_decomposition[key] += 1
        accepted_correct_route += int(kind != "global_unknown" and row["parent"] == row["true_parent"])
        logits = row.get("support_evidence", {}).get("parent_logits")
        if logits is not None:
            import math
            if not isinstance(logits, list) or not logits or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in logits):
                raise ValueError("Parent ranking diagnostics require finite logits")
            top2 = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:2]
            top2_correct += int(row["true_parent"] in top2)
            top2_available += 1
    return {"sample_count": len(near), "terminal_decomposition": terminal,
            "support_candidate_decomposition": candidate_decomposition,
            "support_parent_top1_recall": _ratio(candidate_correct, len(near)),
            "support_parent_top2_recall": _ratio(top2_correct, top2_available),
            "conditional_upper_bounds": {
                "fixed_support_candidate_ideal_two_gates": _ratio(candidate_correct, len(near)),
                "fixed_actual_parent_gate_and_route_ideal_leaf_gate": _ratio(accepted_correct_route, len(near)),
                "required_for_near_target": (17 * len(near) + 19) // 20,
                "scope": "Oracle counts for explicitly fixed candidates/gates; not a guarantee of attainable thresholds."}}


def _source_report(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] != "known":
            source = row.get("source") or row.get("species") or "unspecified"
            groups[(row["status"], normalized_name(source))].append(row)
    records = []
    for (status, source), group in sorted(groups.items()):
        records.append(dict(status=status, source=source,
                            source_names=sorted({r.get("source") or r.get("species") or "unspecified" for r in group}),
                            **base._group_report(group, status)))
    macro = {}
    for status in ("intra", "extra"):
        rates = [r["correct_rate"] for r in records if r["status"] == status]
        macro[status] = dict(source_count=len(rates), correct_rate=None if not rates else sum(rates) / len(rates))
    return {"per_source": records, "source_macro": macro}


def diagnose(reference, selected, meta, train=None, evaluate_test=False):
    """Compare content-aligned archived predictions; no fitting is performed."""
    reference, selected = list(reference), list(selected)
    before = _unique(reference, meta, evaluate_test)
    after = _unique(selected, meta, evaluate_test)
    old = {base._digest(r): r for r in before}
    new = {base._digest(r): r for r in after}
    if set(old) != set(new):
        raise ValueError("Reference/selected content identities differ")
    annotation = ("split", "status", "true_parent", "true_leaf", "source", "species")
    for digest in old:
        if any(old[digest].get(k) != new[digest].get(k) for k in annotation):
            raise ValueError("Paired records have inconsistent truth/source annotations")
    known = [h for h in old if old[h]["status"] == "known"]
    near = [h for h in old if old[h]["status"] == "intra"]
    lost_known = [h for h in known if _correct(old[h]) and not _correct(new[h])]
    gained_known = [h for h in known if not _correct(old[h]) and _correct(new[h])]
    path_eligible = [h for h in near if old[h]["candidate_parent"] == old[h]["true_parent"]
                     and old[h]["prediction_type"] != "global_unknown"]
    lost_path = [h for h in path_eligible if new[h]["prediction_type"] == "global_unknown"]
    retained_path = [h for h in path_eligible if new[h]["prediction_type"] != "global_unknown"
                     and new[h]["parent"] == old[h]["true_parent"]]
    leaves = [h for h in old if old[h]["prediction_type"] == "known"]
    changed_leaves = [h for h in leaves if any(old[h].get(k) != new[h].get(k)
                                             for k in ("prediction_type", "parent", "leaf", "output_node"))]
    train_counts = None
    if train is not None:
        train = base.unique_records(list(train))
        if any(r.get("split") != "train" or r.get("status") != "known" for r in train):
            raise ValueError("TRAIN coverage requires known TRAIN only")
        if {base._digest(r) for r in train} & set(old):
            raise ValueError("TRAIN/evaluation content overlap")
        train_counts = Counter(r["true_leaf"] for r in train)
    coverage = []
    for leaf, name in enumerate(meta["leaf_names"]):
        hashes = [h for h in known if old[h]["true_leaf"] == leaf]
        n = len(hashes)
        coverage.append(dict(leaf=leaf, leaf_name=name, parent=meta["leaf_to_parent"][leaf],
            train_count=None if train_counts is None else train_counts[leaf], evaluation_count=n,
            baseline_correct=sum(_correct(old[h]) for h in hashes),
            selected_correct=sum(_correct(new[h]) for h in hashes),
            lost_baseline_correct=sum(h in lost_known for h in hashes),
            evidence_status="not_evaluable" if n == 0 else "insufficient_evidence" if n <= 4 else "observed_only"))
    attribution = defaultdict(lambda: dict(hit_count=0, changed_count=0, known_harm=0, near_parent_path_harm=0,
                                           terminal_gains=0, terminal_losses=0, sources=set(), protected_known_leaves=set()))
    for digest, row in new.items():
        original = old[digest]
        for index in row.get("applied_rule_indices", []):
            record = attribution[str(index)]
            record["hit_count"] += 1
            record["changed_count"] += int(original["output_node"] != row["output_node"])
            record["known_harm"] += int(digest in lost_known)
            record["near_parent_path_harm"] += int(digest in lost_path)
            record["terminal_gains"] += int(not _correct(original) and _correct(row))
            record["terminal_losses"] += int(_correct(original) and not _correct(row))
            if row["status"] != "known":
                record["sources"].add(row.get("source") or row.get("species") or "unspecified")
            elif _correct(original):
                record["protected_known_leaves"].add(row["true_leaf"])
    for record in attribution.values():
        record["sources"] = sorted(record["sources"])
        record["protected_known_leaves"] = sorted(record["protected_known_leaves"])
    return {"schema_version": "taxosafe_failure_modes_v1", "diagnostic_only": True,
            "evaluation_split": "test" if evaluate_test else "development", "test_opened": bool(evaluate_test),
            "fitting_performed": False, "metric_unit": "unique_image_sha256",
            "input_counts": {"reference": len(reference), "selected": len(selected), "unique_images": len(old)},
            "reference": dict(metrics=base.evaluate_records(before, meta), near=_near_report(before), **_source_report(before)),
            "selected": dict(metrics=base.evaluate_records(after, meta), near=_near_report(after), **_source_report(after)),
            "known_leaf_coverage": coverage,
            "paired_risks": {
                "known_new_harm": dict(_ratio(len(lost_known), len(known)), image_sha256=sorted(lost_known)),
                "known_new_gains": dict(_ratio(len(gained_known), len(known)), image_sha256=sorted(gained_known)),
                "near_parent_path_harm": dict(_ratio(len(lost_path), len(near)), image_sha256=sorted(lost_path)),
                "near_parent_path_retention": _ratio(len(retained_path), len(path_eligible)),
                "original_leaf_invariance": dict(passed=not changed_leaves, leaf_count=len(leaves),
                                                 changed_count=len(changed_leaves), image_sha256=sorted(changed_leaves))},
            "rule_attribution": dict(attribution),
            "interpretation": "Attribution records observed rule co-hits; it is not a causal marginal-effect estimate. Empty/small classes do not establish safety. No parameters were selected from this report."}


def _read_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_run(directory, evaluate_test=False, train_scores=None):
    """Read a reference, geometry/local, or parentrisk run without modifying it."""
    directory = Path(directory).resolve()
    router_path = directory / "calibration/router.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    meta = router["meta"]
    baseline_router = router.get("baseline_router", router)
    paths = [router_path]
    if evaluate_test:
        selected_path = directory / "test/predictions.jsonl"
        before_path = directory / "test/baseline_predictions.jsonl"
    else:
        selected_path = directory / "calibration/development_predictions.jsonl"
        before_path = directory / "calibration/baseline_predictions.jsonl"
    if selected_path.exists():
        selected = _read_records(selected_path)
        paths.append(selected_path)
    elif not evaluate_test and router.get("decoder") == "membership":
        selected_path = directory / "calibration/development_scores.jsonl"
        selected = base.apply_router(_read_records(selected_path), router, meta)
        paths.append(selected_path)
    else:
        raise ValueError("Saved selected predictions are missing: " + str(selected_path))
    if before_path.exists():
        before = _read_records(before_path)
        paths.append(before_path)
    else:
        if baseline_router.get("decoder") != "membership":
            raise ValueError("No archived reference predictions or membership baseline router")
        before = base.apply_router(selected, baseline_router, meta)
    train = None
    if train_scores is not None:
        train_scores = Path(train_scores).resolve()
        train = _read_records(train_scores)
        paths.append(train_scores)
    report = diagnose(before, selected, meta, train, evaluate_test)
    report["source_file_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report["validation_scope"] = "archived_predictions_conditional_on_frozen_reference"
    report["independent_model_level_validation"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-scores", type=Path, help="Optional explicit known TRAIN score archive for coverage only")
    parser.add_argument("--evaluate-test", action="store_true", help="Audit frozen TEST outputs; never fit/select")
    parser.add_argument("--output", type=Path, help="Optional NEW diagnostic JSON file outside the source run")
    args = parser.parse_args()
    if args.output is not None:
        output, source = args.output.resolve(), args.run_dir.resolve()
        if output == source or source in output.parents or output.exists():
            parser.error("Use a new diagnostic output file outside the source run")
    report = audit_run(args.run_dir, args.evaluate_test, args.train_scores)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        print(str(args.output))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
