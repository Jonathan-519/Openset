"""Audit the completed three-seed balanced-v3 development suite on CPU."""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_taxosafe_v1 import diagnose, read, records, root_signature


FIELDS = [
    "intra_macro_parent_species_auroc", "intra_cfr", "intra_oser",
    "known_end_to_end_leaf_accuracy", "known_leaf_coverage",
    "open_world_leaf_precision", "extra_far", "extra_auroc", "drta",
]


def review(suite_dir):
    root = Path(suite_dir).resolve()
    base = diagnose(root)
    rows = []
    root_checks = 0
    for seed in (1, 2, 3):
        folder = root / ("seed_" + str(seed))
        plan = read(folder / "plan.json")
        if plan.get("test_profiles") != "all" or plan.get("primary_profile") != "balanced":
            raise ValueError("Expected a frozen balanced-v3 all-profile plan")
        artifact = folder / "artifacts/full/test"
        coverage = records(artifact / "coverage/predictions.jsonl")
        balanced = records(artifact / "balanced/predictions.jsonl")
        if root_signature(coverage) != root_signature(balanced):
            raise ValueError("Balanced profile changed the frozen coverage root")
        for left, right in zip(coverage, balanced):
            for field in ("candidate_parent", "candidate_leaf", "child_knownness_score"):
                if left[field] != right[field]:
                    raise ValueError("Balanced profile changed score/routing: " + field)
            root_checks += 1
        summary = read(folder / "summary.json")
        for row in summary["rows"]:
            if row["method"] in ("full", "matched_calibration_v4"):
                rows.append({"seed": seed, **row})
    return {"purpose": "Read-only balanced-v3 development audit; no fitting or selection",
            "base_audit": base["audit"], "balanced_root_and_score_checks": root_checks,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.suite_dir).resolve(), Path(args.output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Write diagnostics outside the frozen suite")
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    report = review(source)
    output.mkdir(parents=True)
    (output / "diagnosis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "balanced_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    grouped = collections.defaultdict(list)
    for row in report["rows"]:
        grouped[row["method"], row["profile"]].append(row)
    lines = ["Balanced-v3 development results: three-seed mean +/- sample SD in percent.", "",
             "|method|profile|" + "|".join(FIELDS) + "|",
             "|---|---|" + "---:|" * len(FIELDS)]
    for (method, profile), values in sorted(grouped.items()):
        cells = []
        for field in FIELDS:
            samples = [100 * row[field] for row in values]
            cells.append("{:.2f} +/- {:.2f}".format(
                statistics.mean(samples), statistics.stdev(samples)))
        lines.append("|" + "|".join([method, profile] + cells) + "|")
    (output / "balanced_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Read-only balanced-v3 diagnosis complete:", output)
    print("Verified supplied outputs:", report["base_audit"]["present_outputs_verified"])
    print("Balanced root/score row checks:", report["balanced_root_and_score_checks"])


if __name__ == "__main__":
    main()
