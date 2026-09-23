"""Freeze and execute TaxoSafe-RS experiments, then collect all internal controls.

plan needs only PyYAML; run requires CUDA and the prepared project. Each seed
has its own training run. Test results never select a variant or a profile.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_taxosafe import (DEFAULT_CONFIG, check_data, dependency_report,
                                   ensure_bpe, file_hash)

TEMPLATE = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_residual_clean_v1.yml"
VARIANTS = {"full": {}, "no_parent_residual": {"alphas": [0.0]},
            "no_local_scaling": {"local_scaling": False},
            "fixed_metric": {"alphas": [0.5], "ks": [3]}}
SCRIPTS = {"memory": "build_taxosafe_memory.py", "calibrate": "calibrate_taxosafe_visual.py",
           "test": "test_taxosafe_visual.py"}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def portable(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_plan(config, suite, seed=None, trial=None, run_dir=None):
    source, suite = resolve(config), resolve(suite)
    cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if cfg.get("data", {}).get("split_revision") != "exact-content-deduplicated-v1":
        raise ValueError("This suite requires the clean_v1 training protocol")
    if run_dir and seed is not None:
        raise ValueError("--seed cannot change an existing checkpoint's training configuration")
    seed = int(cfg["seed"] if seed is None else seed)
    trial = str(seed if trial is None else trial)
    if Path(trial).name != trial or trial in (".", ".."):
        raise ValueError("trial must be a directory name")
    if run_dir:
        run = resolve(run_dir).resolve()
        archive = source.resolve()
        if archive.parent != run or not (run / "ckpt/best.pth").is_file():
            raise ValueError("To reuse a run, --training-config must be its archived YAML and ckpt/best.pth must exist")
        content = source.read_bytes()  # Preserve BOM/CRLF for exact archive provenance.
    else:
        cfg["seed"] = seed
        cfg["data"]["seed"] = seed
        cfg["data"]["sampler"].update(seed=seed, holdout_seed=seed)
        cfg["open_treecut"]["holdout_seed"] = seed
        cfg["exp"] = cfg["exp"] + "/rs_suite"
        run = ROOT / "runs" / cfg["data"]["name"] / cfg["model"]["arch"] / cfg["exp"] / ("trial_" + trial)
        archive = run / "training.yml"
        content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode("utf-8")
        if run.exists() and any(run.iterdir()):
            raise FileExistsError("Training run already contains files; reuse its archived YAML or choose a new trial: {}".format(run))
    # Never alter an existing plan, even if it has not been executed yet.
    suite.mkdir(parents=True, exist_ok=False)
    training = suite / "training.yml"
    training.write_bytes(content)
    extension = yaml.safe_load(resolve(TEMPLATE).read_text(encoding="utf-8"))
    extension["base_config"] = portable(archive)
    extension["require_archived_training_config"] = True
    variants = {}
    for name, override in VARIANTS.items():
        value = copy.deepcopy(extension)
        value["visual_support"]["residual"].update(override)
        path = suite / "configs" / (name + ".yml")
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        variants[name] = {"config": portable(path), "artifacts": portable(suite / "artifacts" / name)}
    inputs = {source, training, resolve(TEMPLATE), resolve(cfg["data"]["hierarchy"])}
    inputs.update(resolve(cfg["data"][key]) for key in (
        "train", "val_known", "val_intra", "val_extra",
        "test_known", "test_intra", "test_extra"))
    # OE is an optional training input. Do not require or freeze an oe_train
    # manifest when the configured training objective does not consume it.
    if float(cfg.get("loss", {}).get("lambda_oe", 0.0)) > 0.0:
        oe_train = cfg.get("data", {}).get("oe_train")
        if not oe_train:
            raise ValueError(
                "loss.lambda_oe > 0 requires data.oe_train"
            )
        inputs.add(resolve(oe_train))
    for directory in ("models", "loader", "losses", "optim", "taxosafe_visual"):
        inputs.update((ROOT / directory).rglob("*.py"))
    inputs.update(ROOT / name for name in (
        "train_taxosafe.py", "engine_taxosafe.py", "utils.py", "metrics.py", "taxosafe_episode.py",
        "build_taxosafe_memory.py", "calibrate_taxosafe_visual.py", "test_taxosafe_visual.py",
        "tools/run_taxosafe_suite.py", "tools/prepare_taxosafe.py"))
    inputs.update(resolve(item["config"]) for item in variants.values())
    if run_dir:
        inputs.add(run / "ckpt/best.pth")
    plan = {"schema_version": 1, "seed": seed, "trial": trial,
            "suite": portable(suite), "run_dir": portable(run), "training_config": portable(training),
            "archived_training_config": portable(archive), "reuse_checkpoint": bool(run_dir),
            "variants": variants, "primary_profile": extension["visual_support"]["primary_profile"],
            "inputs_sha256": {portable(p): file_hash(p) for p in sorted(inputs)},
            "protocol": "Freeze all four variants before test. Paired v4 and matched-calibration v4 are emitted by full. No test-based selection."}
    dump(suite / "plan.json", plan)
    return plan


def stage_outputs(stage, folder, profiles="both"):
    if stage == "memory":
        names = ["memory.npz", "memory.json", "residual_state.json"]
    elif stage == "calibrate":
        names = ["calibration.json", "validation_report.json", "validation_scores.jsonl"]
    else:
        names = ["test/comparison.json"]
        profile_names = (["coverage", "risk"] if profiles == "both" else
                         ["coverage", "balanced", "risk"] if profiles == "all" else
                         [profiles])
        for profile in profile_names:
            names += ["test/{}/{}".format(profile, name) for name in (
                "metrics.json", "predictions.jsonl", "baseline_v4_metrics.json", "baseline_v4_predictions.jsonl",
                "matched_v4_metrics.json", "matched_v4_predictions.jsonl")]
    return [portable(resolve(folder) / name) for name in names]


def steps(plan, stage="all"):
    out = []
    if stage in ("all", "train") and not plan["reuse_checkpoint"]:
        out.append({"id": "train", "stage": "train", "argv": ["train_taxosafe.py", "--config", plan["training_config"], "--trial", plan["trial"]],
                    "outputs": [portable(resolve(plan["run_dir"]) / "ckpt/best.pth"), plan["archived_training_config"]]})
    if stage == "train" and plan["reuse_checkpoint"]:
        raise ValueError("This plan reuses an existing checkpoint; it has no training step")
    for current in ("memory", "calibrate", "test"):
        if stage not in ("all", current):
            continue
        for name, item in plan["variants"].items():
            test_profiles = item.get("test_profiles", plan.get("test_profiles", "both"))
            argv = [SCRIPTS[current], "--config", item["config"], "--trial", plan["trial"],
                    "--run-dir", plan["run_dir"], "--artifact-dir", item["artifacts"]]
            if current == "test":
                argv += ["--profiles", test_profiles]
            out.append({"id": name + "_" + current, "stage": current, "argv": argv,
                        "outputs": stage_outputs(current, item["artifacts"], test_profiles)})
    return out


def verify_inputs(plan):
    for path, expected in plan["inputs_sha256"].items():
        if not resolve(path).is_file() or file_hash(resolve(path)) != expected:
            raise ValueError("Frozen input changed: {}. Create a new suite; do not change a tested plan.".format(path))


def receipt_path(plan, step):
    return resolve(plan["suite"]) / "receipts" / (step["id"] + ".json")


def verify_receipt(plan, step):
    path = receipt_path(plan, step)
    if not path.is_file():
        return False
    receipt = read(path)
    if receipt["plan_sha256"] != file_hash(resolve(plan["suite"]) / "plan.json"):
        raise ValueError("Receipt/plan mismatch: " + step["id"])
    if set(receipt["outputs_sha256"]) != set(step["outputs"]):
        raise ValueError("Receipt output list mismatch: " + step["id"])
    for output, expected in receipt["outputs_sha256"].items():
        if not resolve(output).is_file() or file_hash(resolve(output)) != expected:
            raise ValueError("Completed output changed/missing: " + output)
    return True


def verify_prerequisites(plan, step):
    needed = []
    if step["stage"] != "train" and not plan["reuse_checkpoint"]:
        needed += steps(plan, "train")
    for earlier in ("memory", "calibrate"):
        if ((step["stage"] == "calibrate" and earlier == "memory")
                or step["stage"] == "test"):
            name = step["id"].rsplit("_", 1)[0]
            needed += [s for s in steps(plan, earlier) if s["id"] == name + "_" + earlier]
    for previous in needed:
        if not verify_receipt(plan, previous):
            raise FileNotFoundError("Incomplete prerequisite: " + previous["id"])


def execute(plan, stage, resume=False, dry_run=False):
    verify_inputs(plan)
    pending = []
    for step in steps(plan, stage):
        if resume and verify_receipt(plan, step):
            verify_prerequisites(plan, step)
            print("Verified complete: " + step["id"], flush=True)
            continue
        existing = [p for p in step["outputs"] if resolve(p).exists()]
        if existing:
            raise FileExistsError("Existing outputs without reusable receipt: {}. Use --resume for completed stages; inspect partial outputs manually.".format(existing))
        pending.append(step)
    if dry_run:
        for step in pending:
            print(shlex.join([sys.executable] + step["argv"]))
        return
    if not pending:
        return
    ensure_bpe()
    errors = dependency_report(require_cuda=True)["errors"]
    if errors:
        raise RuntimeError("\n".join(errors))
    os.chdir(ROOT)
    for step in pending:
        verify_prerequisites(plan, step)
        if step["stage"] == "train":
            run = resolve(plan["run_dir"])
            if run.exists() and any(run.iterdir()):
                raise FileExistsError("Training would reuse an existing run: " + str(run))
            check_data(plan["training_config"])
        else:
            archive = resolve(plan["archived_training_config"])
            if not archive.is_file() or file_hash(archive) != file_hash(resolve(plan["training_config"])):
                raise ValueError("Training archive differs from the frozen plan; do not mix checkpoints and configurations")
        print("Running " + step["id"], flush=True)
        log = resolve(plan["suite"]) / "logs" / (step["id"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen([sys.executable, "-u"] + step["argv"], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError("{} failed with exit {}; see {}".format(step["id"], returncode, log))
        dump(receipt_path(plan, step), {"plan_sha256": file_hash(resolve(plan["suite"]) / "plan.json"),
             "outputs_sha256": {p: file_hash(resolve(p)) for p in step["outputs"]}})


def root_records(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["status"], row["path"], row["image_sha256"])
        if key in result:
            raise ValueError("Repeated test image in predictions: " + str(key))
        result[key] = [row[k] for k in ("root_knownness_score", "root_gate_margin", "candidate_parent", "parent")]
        result[key].append(row["prediction_type"] == "global_unknown")
    return result


def summarize(plan):
    verify_inputs(plan)
    for step in steps(plan, "test"):
        verify_prerequisites(plan, step)
        if not verify_receipt(plan, step):
            raise FileNotFoundError("Incomplete test stage: " + step["id"])
    full = resolve(plan["variants"]["full"]["artifacts"])
    reports = {name: read(resolve(item["artifacts"]) / "test/comparison.json")
               for name, item in plan["variants"].items()}
    rows = []
    requested = plan.get("test_profiles", "both")
    profile_names = (["coverage", "risk"] if requested == "both" else
                     ["coverage", "balanced", "risk"] if requested == "all" else
                     [requested])
    for profile in profile_names:
        reference = root_records(full / "test" / profile / "baseline_v4_predictions.jsonl")
        checkpoints = set()
        for name, item in plan["variants"].items():
            folder = resolve(item["artifacts"]) / "test" / profile
            if root_records(folder / "predictions.jsonl") != reference:
                raise ValueError("Frozen-root comparison failed: {} {}".format(name, profile))
            checkpoints.add(read(folder / "metrics.json")["metadata"]["checkpoint_sha256"])
        if len(checkpoints) != 1:
            raise ValueError("Variants did not use the same checkpoint")
        for name, values in (
            ("paired_v4", reports["full"]["paired_baseline_v4"][profile]),
            ("matched_calibration_v4", reports["full"]["matched_calibration_v4"][profile]),
            *((name, report["profiles"][profile]) for name, report in reports.items())):
            rows.append({"method": name, "profile": profile, **values})
    summary = {"seed": plan["seed"], "primary_profile": plan["primary_profile"],
               "root_predictions_identical_across_variants": True, "rows": rows,
               "note": "Single-seed internal comparisons; no method selection, significance claim or external literature baseline."}
    suite = resolve(plan["suite"])
    dump(suite / "summary.json", summary)
    columns = ["method", "profile", "intra_macro_parent_species_auroc", "intra_cfr", "intra_oser",
               "known_end_to_end_leaf_accuracy", "known_leaf_coverage", "extra_far", "extra_auroc", "drta"]
    lines = ["TaxoSafe-RS seed {}. Primary profile: {}. Values are fractions, not percentages.".format(plan["seed"], plan["primary_profile"]),
             "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = ["NA" if row.get(c) is None else ("{:.4f}".format(row[c]) if isinstance(row[c], float) else str(row[c])) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    (suite / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Freeze configs before any test evaluation; no GPU needed")
    plan_parser.add_argument("--training-config", default=DEFAULT_CONFIG)
    plan_parser.add_argument("--suite-dir", required=True)
    plan_parser.add_argument("--seed", type=int)
    plan_parser.add_argument("--trial")
    plan_parser.add_argument("--run-dir", help="Reuse a trained run; requires its archived --training-config")
    run_parser = sub.add_parser("run", help="Run frozen steps with output verification")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--stage", choices=["all", "train", "memory", "calibrate", "test"], default="all")
    run_parser.add_argument("--resume", action="store_true", help="Skip only completed steps with matching output hashes")
    run_parser.add_argument("--dry-run", action="store_true")
    report_parser = sub.add_parser("summarize", help="Report all methods, preserving the preregistered primary profile")
    report_parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            plan = make_plan(args.training_config, args.suite_dir, args.seed, args.trial, args.run_dir)
            for step in steps(plan):
                print(shlex.join(["python"] + step["argv"]))
            print("Frozen plan: " + str(resolve(plan["suite"]) / "plan.json"))
        else:
            plan = read(resolve(args.plan))
            if args.command == "run":
                execute(plan, args.stage, args.resume, args.dry_run)
            else:
                summarize(plan)
                print("Summary: " + str(resolve(plan["suite"]) / "summary.md"))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print("STOP: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
