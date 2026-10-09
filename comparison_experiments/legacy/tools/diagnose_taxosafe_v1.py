"""Audit completed v1 artifacts and cross the two ALREADY fitted gates on CPU.

This is a post-hoc development diagnostic, not a new confirmatory experiment.
No threshold is fitted, no model is trained, and no best test row is selected.
"""
import argparse
import collections
import copy
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxosafe_visual.pipeline import _metrics, _summary
from tools.prepare_taxosafe import file_hash
from tools.run_taxosafe_suite import steps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(rows):
    result = {}
    for row in rows:
        key = (row["status"], row["path"], row["image_sha256"])
        if key in result:
            raise ValueError("Repeated prediction identity: {}".format(key))
        result[key] = row
    return result


def combine_gates(root_rows, child_rows, label):
    """Only stored predictions/scores determine decisions; truth is unused."""
    left, right = indexed(root_rows), indexed(child_rows)
    if left.keys() != right.keys():
        raise ValueError("Profiles contain different test images")
    output = []
    for key, root in left.items():
        child = right[key]
        for field in ("candidate_parent", "candidate_leaf", "root_knownness_score", "child_knownness_score"):
            if root[field] != child[field]:
                raise ValueError("Profile scores/routing differ: " + field)
        row = copy.deepcopy(root)
        row["profile"] = label
        row["residual_child_threshold"] = child["residual_child_threshold"]
        row["child_gate_margin"] = row["child_knownness_score"] - row["residual_child_threshold"]
        if root["prediction_type"] != "global_unknown":
            accept = row["child_gate_margin"] >= 0
            row["prediction_type"] = "known" if accept else "intra_unknown"
            row["parent"] = row["candidate_parent"]
            row["parent_name"] = row["candidate_parent_name"]
            row["leaf"] = row["candidate_leaf"] if accept else None
            row["leaf_name"] = row["candidate_leaf_name"] if accept else None
        output.append(row)
    return output


def root_signature(rows):
    return {key: (r["root_knownness_score"], r["root_gate_margin"], r["candidate_parent"],
                  r["parent"], r["prediction_type"] == "global_unknown")
            for key, r in indexed(rows).items()}


def diagnose(suite_root):
    root = Path(suite_root).resolve()
    if {p.name for p in root.glob("seed_*") if p.is_dir()} != {"seed_1", "seed_2", "seed_3"}:
        raise ValueError("Expected completed seed_1, seed_2 and seed_3")
    audit = {"present_outputs_verified": 0, "unavailable_outputs": [], "input_hashes": {},
             "root_prediction_checks": 0, "checkpoint_hashes": {}, "split_counts": {}}
    summaries, drivers, selections = [], [], []

    def local_path(name):
        p = Path(name)
        for i, part in enumerate(p.parts):
            if part == root.name:
                return root.joinpath(*p.parts[i + 1:])
        return ROOT / p if not p.is_absolute() else p

    for seed in (1, 2, 3):
        folder = root / ("seed_" + str(seed))
        plan = read(folder / "plan.json")
        if plan["seed"] != seed:
            raise ValueError("Seed/plan mismatch")
        for step in steps(plan):
            receipt_file = folder / "receipts" / (step["id"] + ".json")
            receipt = read(receipt_file)
            if set(receipt["outputs_sha256"]) != set(step["outputs"]):
                raise ValueError("Receipt output list mismatch: " + str(receipt_file))
            if receipt["plan_sha256"] != file_hash(folder / "plan.json"):
                raise ValueError("Receipt/plan hash mismatch: " + str(receipt_file))
            for name, expected in receipt["outputs_sha256"].items():
                path = local_path(name)
                if path.is_file():
                    if file_hash(path) != expected:
                        raise ValueError("Receipt output bytes differ: " + str(path))
                    audit["present_outputs_verified"] += 1
                else:
                    audit["unavailable_outputs"].append(name)
        checkpoints = set()
        reference_roots = {}
        for variant in plan["variants"]:
            art = folder / "artifacts" / variant
            calibration, memory, state = (read(art / n) for n in
                                         ("calibration.json", "memory.json", "residual_state.json"))
            meta = calibration["taxonomy"]
            if memory["checkpoint_sha256"] != calibration["metadata"]["checkpoint_sha256"]:
                raise ValueError("Calibration/memory checkpoint mismatch")
            if state["memory_sha256"] != memory["memory_sha256"]:
                raise ValueError("State/memory mismatch")
            if calibration["residual_state_sha256"] != file_hash(art / "residual_state.json"):
                raise ValueError("Calibration/state mismatch")
            checkpoints.add(memory["checkpoint_sha256"])
            rows = {profile: records(art / "test" / profile / "predictions.jsonl")
                    for profile in ("coverage", "risk")}
            for profile in rows:
                baseline = records(art / "test" / profile / "baseline_v4_predictions.jsonl")
                if root_signature(rows[profile]) != root_signature(baseline):
                    raise ValueError("Frozen-root invariant failed")
                signature = root_signature(baseline)
                if profile in reference_roots and reference_roots[profile] != signature:
                    raise ValueError("Root predictions differ across variants")
                reference_roots[profile] = signature
                audit["root_prediction_checks"] += len(rows[profile])
            train = set(memory["all_train_image_hashes"])
            validation = set(calibration["metadata"]["validation_image_hashes"])
            fit = set(calibration["metadata"]["fit_image_hashes"])
            held = set(calibration["metadata"]["threshold_image_hashes"])
            test = {r["image_sha256"] for r in rows["coverage"]}
            if (train & validation or train & test or validation & test or fit & held
                    or fit | held != validation):
                raise ValueError("Image-hash split separation failed")
            audit["split_counts"][str(seed)] = dict(train=len(train), validation=len(validation),
                                                    test=len(test), fit=len(fit), held=len(held))
            selections.append({"seed": seed, "variant": variant, **state["state"]["selection"]["chosen"],
                               "local_scaling": state["state"]["local_scaling"]})
            validation_rows = [r for r in records(art / "validation_scores.jsonl")
                               if r["profile"] == "coverage" and r["validation_partition"] == "threshold_calibration"]
            residual_calibration = calibration["residual_calibration"]
            for parent, name in enumerate(meta["parent_names"]):
                coverage_value = residual_calibration["profiles"]["coverage"][str(parent)]
                if isinstance(coverage_value, dict):
                    branch = residual_calibration["branches"][name]
                    counts = branch.get("known_counts_correct_leaf", {})
                    for leaf_id, tau in sorted(
                            coverage_value["by_leaf"].items(), key=lambda item: int(item[0])):
                        leaf = meta["leaf_names"][int(leaf_id)]
                        drivers.append(dict(
                            seed=seed, variant=variant, parent=name,
                            controlling_leaf=None, leaf_specific_threshold=leaf,
                            calibration_samples=int(counts.get(leaf, 0)),
                            threshold=float(tau),
                            threshold_mode=residual_calibration.get("threshold_mode")))
                    continue
                groups = collections.defaultdict(list)
                for row in validation_rows:
                    if row["status"] == "known" and row["true_parent"] == parent == row["candidate_parent"]:
                        groups[row["true_leaf_name"]].append(row["child_knownness_score"])
                rejection = calibration["metadata"]["settings"].get("child_known_rejection", .1)
                boundaries = sorted((sorted(values)[int(rejection * len(values))], leaf, len(values))
                                    for leaf, values in groups.items())
                if boundaries:
                    tau, leaf, count = boundaries[0]
                    drivers.append(dict(seed=seed, variant=variant, parent=name, controlling_leaf=leaf,
                                        leaf_specific_threshold=None, calibration_samples=count,
                                        threshold=tau, threshold_mode="branch_min"))
            for rp in ("coverage", "risk"):
                for cp in ("coverage", "risk"):
                    name = "root_{}__child_{}".format(rp, cp)
                    crossed = combine_gates(rows[rp], rows[cp], name)
                    if root_signature(crossed) != root_signature(rows[rp]):
                        raise AssertionError("Crossed gates changed selected root")
                    m = _metrics(crossed)
                    intra = [r for r in crossed if r["status"] == "intra"]
                    opportunity = sum(r["prediction_type"] != "global_unknown" and
                                      r["candidate_parent"] == r["true_parent"] for r in intra) / len(intra)
                    summaries.append({"seed": seed, "method": variant, "profile": name, **_summary(m),
                                      "intra_global_rejection": m["intra"]["intra_global_rejection_rate"],
                                      "known_global_rejection": m["known"]["known_global_rejection_rate"],
                                      "cfr_upper_bound_with_frozen_root": opportunity})
        if len(checkpoints) != 1:
            raise ValueError("Variants did not use the same checkpoint")
        audit["checkpoint_hashes"][str(seed)] = next(iter(checkpoints))
    if len(set(audit["checkpoint_hashes"].values())) != 3:
        raise ValueError("Different seeds unexpectedly share checkpoint bytes")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".yml"):
            audit["input_hashes"][str(path.relative_to(root))] = file_hash(path)
    return {"purpose": "Post-hoc development diagnosis; no fitting or test-based selection",
            "audit": audit, "crossed_gates": summaries, "threshold_drivers": drivers,
            "training_selection": selections}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    output = Path(args.output_dir).resolve()
    source = Path(args.suite_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Write diagnostic output outside the original frozen suite")
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    report = diagnose(source)
    output.mkdir(parents=True)
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "crossed_gates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["crossed_gates"][0]))
        writer.writeheader(); writer.writerows(report["crossed_gates"])
    fields = ["intra_cfr", "intra_oser", "known_end_to_end_leaf_accuracy", "known_leaf_coverage",
              "extra_far", "drta", "intra_global_rejection", "cfr_upper_bound_with_frozen_root"]
    groups = collections.defaultdict(list)
    for row in report["crossed_gates"]:
        groups[row["method"], row["profile"]].append(row)
    lines = ["Post-hoc crossed gates: three-seed mean +/- sample SD in percent. Not a new confirmatory result.", "",
             "|method|profile|" + "|".join(fields) + "|", "|---|---|" + "---:|" * len(fields)]
    for (method, profile), rows in sorted(groups.items()):
        values = ["{:.2f} +/- {:.2f}".format(statistics.mean([r[f] * 100 for r in rows]),
                                            statistics.stdev([r[f] * 100 for r in rows])) for f in fields]
        lines.append("|" + "|".join([method, profile] + values) + "|")
    (output / "crossed_gates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Read-only diagnosis complete: " + str(output))
    print("Verified supplied outputs:", report["audit"]["present_outputs_verified"])
    print("Outputs not supplied/available:", len(report["audit"]["unavailable_outputs"]))
    print("Root row checks:", report["audit"]["root_prediction_checks"])


if __name__ == "__main__":
    main()
