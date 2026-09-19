"""Frozen v4 caches -> train-only verifier -> known-only calibration -> explicit test.

Only new files are installed, so pre-existing v4 frozen hashes remain valid.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import statistics
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from taxosafe_visual.runtime import resolve, read_json, write_json, sha256
from tools import run_taxosafe_hier_v4 as old_runner

CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_witness_v5.yml"
STAGES = ("fit", "calibrate", "test")


def portable(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def make_plan(source_suite, suite, seed, config=CONFIG):
    source_path = resolve(source_suite) / ("seed_" + str(seed)) / "plan.json"
    if not source_path.is_file():
        raise FileNotFoundError("Need completed v4 caches; missing " + str(source_path))
    src = read_json(source_path)
    old_runner.verify_inputs(src)
    if src["seed"] != seed:
        raise ValueError("Source seed mismatch")
    paths = {source_path, resolve(config)}
    paths.update(resolve(p) for p in src["inputs_sha256"])
    for stage in ("cache_train", "train", "cache_val", "calibrate"):
        if not old_runner.verified(src, stage):
            raise ValueError("Complete v4 " + stage + " first")
        paths.add(resolve(src["suite"]) / "receipts" / (stage + ".json"))
        paths.update(resolve(src["suite"]) / p for p in old_runner.outputs(src, stage))
    paths.update((ROOT / "taxosafe_witness").glob("*.py"))
    paths.add(Path(__file__))
    paths.add(ROOT / "tools/run_taxosafe_witness_v5.sh")
    settings = yaml.safe_load(resolve(config).read_text())
    from taxosafe_witness.core import VARIANTS
    if settings["variants"] != list(VARIANTS) or settings["classifier"] != "frozen_v4_identity" or settings["primary_variant"] != "full":
        raise ValueError("Keep the fixed classifier and all declared variants")
    folder = resolve(suite) / ("seed_" + str(seed))
    plan = {"schema": 1, "seed": seed, "suite": portable(folder), "source_plan": portable(source_path),
            "settings": settings, "taxonomy": src["taxonomy"],
            "inputs_sha256": {portable(p): sha256(p) for p in sorted(paths)},
            "test_loaded": False, "development_only": True}
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", plan)
    return folder / "plan.json"


def outputs(plan, stage):
    if stage == "fit":
        return ["models.json", "training_report.json"]
    if stage == "calibrate":
        return ["calibration.json", "validation_report.json", "validation_evidence.jsonl"]
    if stage == "test":
        return ["summary.json"] + ["test/{}/{}".format(v, f) for v in plan["settings"]["variants"]
                                  for f in ("metrics.json", "predictions.jsonl")]
    raise ValueError("Unknown stage")


def verify_inputs(plan):
    for name, digest in plan["inputs_sha256"].items():
        if not resolve(name).is_file() or sha256(resolve(name)) != digest:
            raise ValueError("Frozen input changed: " + name + "; retain this run and create a new suite")


def verified(plan, stage):
    folder = resolve(plan["suite"])
    path = folder / "receipts" / (stage + ".json")
    if not path.exists():
        return False
    r = read_json(path)
    if r["plan_sha256"] != sha256(folder / "plan.json") or set(r["outputs"]) != set(outputs(plan, stage)):
        raise ValueError("Receipt/plan mismatch: " + stage)
    for name, digest in r["outputs"].items():
        if not (folder / name).is_file() or sha256(folder / name) != digest:
            raise ValueError("Completed output changed: " + name)
    for name, digest in r.get("stage_inputs", {}).items():
        if not resolve(name).is_file() or sha256(resolve(name)) != digest:
            raise ValueError("Test input changed: " + name)
    return True


def test_inputs(plan):
    src = read_json(resolve(plan["source_plan"]))
    if not old_runner.verified(src, "cache_test"):
        raise ValueError("Run the original v4 test cache stage first; v5 never extracts images")
    paths = [resolve(src["suite"]) / p for p in old_runner.outputs(src, "cache_test")]
    previous = resolve(src["suite"]) / "artifacts/identity/test/coverage/predictions.jsonl"
    if previous.exists():
        if not old_runner.verified(src, "test"):
            raise ValueError("Original v4 test outputs are not verified")
        paths.append(previous)
    return {portable(p): sha256(p) for p in paths}


def run(plan, selection, resume=False, dry_run=False):
    verify_inputs(plan)
    selected = STAGES[:2] if selection == "develop" else (selection,)
    folder = resolve(plan["suite"])
    for stage in selected:
        earlier = STAGES[:STAGES.index(stage)]
        if resume and verified(plan, stage):
            if not all(verified(plan, p) for p in earlier):
                raise ValueError("Missing completed prerequisite")
            print("Verified complete:", stage, flush=True)
            continue
        if any((folder / p).exists() for p in outputs(plan, stage)):
            raise FileExistsError("Partial/existing output: " + stage + "; preserve this suite and use a new name")
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "_step", "--plan", str(folder / "plan.json"), "--stage", stage]
        if dry_run:
            import shlex
            print(shlex.join(argv)); continue
        if not all(verified(plan, p) for p in earlier):
            raise ValueError("Complete earlier stages first")
        extra = test_inputs(plan) if stage == "test" else {}
        write_json(folder / "pending" / (stage + ".json"), {"stage_inputs": extra})
        log = folder / "logs" / (stage + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[key] = str(plan["settings"]["threads"])
        with log.open("a") as stream:
            with subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True); stream.write(line)
                status = process.wait()
        if status:
            raise RuntimeError("Stage failed; see " + str(log))
        for name, digest in extra.items():
            if sha256(resolve(name)) != digest:
                raise ValueError("Stage input changed during execution")
        verify_inputs(plan)
        write_json(folder / "receipts" / (stage + ".json"), {"plan_sha256": sha256(folder / "plan.json"),
                   "outputs": {p: sha256(folder / p) for p in outputs(plan, stage)}, "stage_inputs": extra})


def summarize(suite):
    rows = []
    for seed in (1, 2, 3):
        folder = resolve(suite) / ("seed_" + str(seed))
        plan = read_json(folder / "plan.json")
        verify_inputs(plan)
        if not all(verified(plan, s) for s in STAGES):
            raise ValueError("Need three fully completed seeds")
        summary = read_json(folder / "summary.json")
        rows.extend({"seed": seed, **r} for r in summary["rows"])
        rows.extend({"seed": seed, "method": "legacy_identity_" + p, **r} for p, r in summary["legacy_v4"].items())
    fields = ["closed_routed_accuracy", "closed_routed_macro_accuracy", "known_end_to_end_leaf_accuracy",
              "intra_macro_parent_species_auroc", "intra_oser", "intra_cfr", "extra_final_known_false_acceptance"]
    lines = ["Development results, mean +/- sample SD (%).", "", "|method|" + "|".join(fields) + "|", "|---|" + "---:|" * len(fields)]
    for variant in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == variant]
        vals = []
        for key in fields:
            a = [100 * r[key] for r in rs if r[key] is not None]
            vals.append("{:.2f} +/- {:.2f}".format(statistics.mean(a), statistics.stdev(a)) if len(a) == 3 else "NA")
        lines.append("|" + "|".join([variant] + vals) + "|")
    write_json(resolve(suite) / "aggregate.json", {"rows": rows})
    (resolve(suite) / "aggregate.md").write_text("\n".join(lines) + "\n")
    print("Summary:", resolve(suite) / "aggregate.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--source-suite", default="runs/taxosafe_hier_v4_dev")
    p.add_argument("--suite-dir", default="runs/taxosafe_witness_v5_dev")
    p.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--config", default=CONFIG)
    for name in ("run", "_step"):
        p = sub.add_parser(name); p.add_argument("--plan", required=True)
        p.add_argument("--stage", choices=STAGES + ("develop",), default="develop")
        p.add_argument("--resume", action="store_true"); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("summarize"); p.add_argument("--suite-dir", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "plan":
        print("Frozen v5 plan:", make_plan(args.source_suite, args.suite_dir, args.seed, args.config))
    elif args.command == "summarize":
        summarize(args.suite_dir)
    else:
        plan = read_json(resolve(args.plan))
        if args.command == "run":
            run(plan, args.stage, args.resume, args.dry_run)
        else:
            verify_inputs(plan)
            from taxosafe_witness import pipeline
            if args.stage not in STAGES:
                raise ValueError("_step requires one stage")
            if args.stage == "test":
                pending = read_json(resolve(plan["suite"]) / "pending/test.json")["stage_inputs"]
                if pending != test_inputs(plan):
                    raise ValueError("Test sources changed before execution")
            getattr(pipeline, args.stage)(plan)


if __name__ == "__main__":
    main()
