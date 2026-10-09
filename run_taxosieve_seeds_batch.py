#!/usr/bin/env python3
"""Run isolated TaxoSieve seed repetitions without altering the original source.

Requires the taxosieve-data-rebuild recipe and its installed Python environment.
Example: python tools/run_taxosieve_seeds.py --seeds 2 3 4 5
Batch ablation: --seeds 8 18 28 38 48 --train-batch-size 24
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import uuid

PACKAGES = ("taxosieve", "taxosafe_support", "models", "loader")
ROOT_FILES = ("run_taxosieve.py", "taxosafe_episode.py", "metrics_open.py")
INPUT_KEYS = ("train", "val_known", "val_intra", "val_extra", "test_known",
              "test_intra", "test_extra", "hierarchy", "known_preparation_audit")
METRICS = {
    "known": "known_end_to_end_leaf_accuracy",
    "near": "intra_correct_fallback_rate",
    "extra": "extra_global_unknown_recall",
    "precision": "open_world_accepted_leaf_precision",
}
RANKING = ("eligible only with complete OOF and four finite metrics; prefer all "
           "four metric targets passed, then maximum minimum metric, near, known, "
           "extra, precision, then lower seed; TEST is not used for selection")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_seeds(seeds):
    seeds = list(seeds)
    if not seeds or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds):
        raise ValueError("Seeds must be integers in [0, 2**31), matching the model validators")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not independent repetitions")
    return seeds


def batch_recipe(ref, train_batch_size=None):
    """Describe an explicit image-batch ablation at a fixed sampled-image budget."""
    data = ref["data"]
    sampler = data["sampler"]
    expected = dict(parents_per_batch=2, species_per_parent=3,
                    images_per_species=2, batches_per_epoch=240)
    if (data.get("batch_size") != 12 or sampler.get("name") != "hierarchical_episode"
            or any(sampler.get(key) != value for key, value in expected.items())):
        raise ValueError("Expected the original image batch 12 and 2x3x2 sampler with 240 batches")
    batch = 12 if train_batch_size is None else train_batch_size
    if type(batch) is not int or batch < 12 or batch % 6 or 2880 % batch:
        raise ValueError("--train-batch-size must be an integer >=12, a multiple of 6, and divide 2880 (for example 12, 24, 48)")
    return dict(train_batch_size=batch, eval_batch_size=data["eval_batch_size"],
        parents_per_batch=2, species_per_parent=3, images_per_species=batch // 6,
        batches_per_epoch=2880 // batch, sampled_images_per_epoch=2880,
        baseline_train_batch_size=12, baseline_batches_per_epoch=240,
        variant="baseline_seed_sweep" if batch == 12 else "image_batch_ablation",
        learning_rates_changed=False, same_optimizer_steps_as_baseline=batch == 12)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Unsupported source version; expected exactly one patch anchor: " + before[:100])
    return text.replace(before, after, 1)


def patch_runtime(runtime):
    """Patch physical copied files; original recipe and all digest checks remain."""
    runtime = Path(runtime)
    path = runtime / "taxosieve/protocol.py"
    text = path.read_text(encoding="utf-8")
    old = '''def effective_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    # This entry point intentionally reproduces one experiment. New ablations
    # belong in comparison_experiments, with a distinct identity and receipt.
    if object_hash(cfg) != object_hash(DEFAULTS):
        raise ValueError("TaxoSieve settings differ from the locked TaxoSieve_v1 recipe")
    return copy.deepcopy(cfg)
'''
    new = '''def validate_seed_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("TaxoSieve configuration must be a mapping")
    result = copy.deepcopy(cfg)
    seeds = []
    for name in ("d05", "d05_calibration", "calibration"):
        if not isinstance(result.get(name), dict):
            raise ValueError("Missing seed-bearing configuration: " + name)
        seed = result[name].get("seed")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Seed must be an integer in [0, 2**31)")
        seeds.append(seed)
        result[name]["seed"] = DEFAULTS[name]["seed"]
    if len(set(seeds)) != 1 or object_hash(result) != object_hash(DEFAULTS):
        raise ValueError("Only a common seed may differ from the locked TaxoSieve_v1 recipe")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG, seed=None):
    cfg = validate_seed_config(yaml.safe_load(resolve(path).read_text(encoding="utf-8")))
    if seed is not None:
        for name in ("d05", "d05_calibration", "calibration"):
            cfg[name]["seed"] = seed
    return validate_seed_config(cfg)
'''
    text = replace_once(text, old, new)
    text = replace_once(text,
        '    value = read_json(regular(directory / "run.json"))\n    if (value.get("schema_version") != SCHEMA or value.get("config") != DEFAULTS',
        '    value = read_json(regular(directory / "run.json"))\n    validate_seed_config(value.get("config"))\n    if (value.get("schema_version") != SCHEMA')
    text = replace_once(text,
        'def initialize(directory, cfg, reference_directory, device, mode="train"):\n',
        'def initialize(directory, cfg, reference_directory, device, mode="train"):\n    cfg = validate_seed_config(cfg)\n')
    path.write_text(text, encoding="utf-8")
    path = runtime / "taxosieve/pipeline.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(text,
        'def train(directory, reference_directory=None, device="cuda", config=p.DEFAULT_CONFIG, resume=False):',
        'def train(directory, reference_directory=None, device="cuda", config=p.DEFAULT_CONFIG, resume=False, seed=None):')
    text = replace_once(text,
        '        if p.effective_config(config) != run["config"]:',
        '        actual_seed = run["config"]["d05"]["seed"] if seed is None else seed\n        if p.effective_config(config, seed=actual_seed) != run["config"]:')
    text = replace_once(text,
        '    else:\n        cfg = p.effective_config(config)\n        reference_directory =',
        '    else:\n        cfg = p.effective_config(config, seed=seed)\n        reference_directory =')
    text = replace_once(text,
        'def preflight(config=p.DEFAULT_CONFIG, reference_directory=None, metadata_only=False):',
        'def preflight(config=p.DEFAULT_CONFIG, reference_directory=None, metadata_only=False, seed=None):')
    text = replace_once(text,
        '    import torch\n    cfg = p.effective_config(config)\n    ref =',
        '    import torch\n    cfg = p.effective_config(config, seed=seed)\n    ref =')
    text = replace_once(text,
        '        child.add_argument("--resume", action="store_true")',
        '        child.add_argument("--resume", action="store_true")\n        child.add_argument("--seed", type=int, default=None)')
    text = replace_once(text,
        '    child.add_argument("--metadata-only", action="store_true", help="Validate locked metadata without opening image bytes")',
        '    child.add_argument("--metadata-only", action="store_true", help="Validate locked metadata without opening image bytes")\n    child.add_argument("--seed", type=int, default=None)')
    text = replace_once(text,
        'result = train(args.run_dir, args.reference_run_dir, args.device, args.config, args.resume)',
        'result = train(args.run_dir, args.reference_run_dir, args.device, args.config, args.resume, seed=args.seed)')
    text = replace_once(text,
        'result = preflight(args.config, args.reference_run_dir, args.metadata_only)',
        'result = preflight(args.config, args.reference_run_dir, args.metadata_only, seed=args.seed)')
    path.write_text(text, encoding="utf-8")
    for name in ("taxosieve/protocol.py", "taxosieve/pipeline.py"):
        compile((runtime / name).read_text(encoding="utf-8"), str(runtime / name), "exec")


def prepare_suite(project_root, seeds=(2, 3, 4, 5), output=None, device="cuda", train_batch_size=None):
    """Prepare an exclusive code snapshot and immutable plan without importing torch."""
    import yaml
    project = Path(project_root).resolve()
    seeds = validate_seeds(seeds)
    if device not in ("cuda", "cpu"):
        raise ValueError("Device must be cuda or cpu")
    files = [Path(name) for name in ROOT_FILES]
    files += [Path("models/bpe_simple_vocab_16e6.txt.gz")]
    for package in PACKAGES:
        members = sorted((project / package).glob("*.py"))
        if not members:
            raise ValueError("Missing source package: " + package)
        files += [p.relative_to(project) for p in members]
    files += [p.relative_to(project) for p in sorted((project / "configs").glob("taxosieve*.yml"))]
    cfg = yaml.safe_load((project / "configs/taxosieve.yml").read_text(encoding="utf-8"))
    if cfg.get("reference_config") != "configs/taxosieve_reference.yml":
        raise ValueError("Expected the rebuilt TaxoSieve reference configuration")
    if any(cfg.get(k, {}).get("seed") != 1 for k in ("d05", "d05_calibration", "calibration")):
        raise ValueError("The source recipe must be the unmodified seed=1 recipe")
    ref = yaml.safe_load((project / cfg["reference_config"]).read_text(encoding="utf-8"))
    recipe = batch_recipe(ref, train_batch_size)
    inputs = {str((project / ref["data"][key]).resolve()) for key in INPUT_KEYS}
    inputs.update(str((project / name).resolve()) for name in ("configs/taxosieve.yml", cfg["reference_config"]))
    input_hashes = {name: digest(name) for name in sorted(inputs)}
    source_hashes = {str(name): digest(project / name) for name in files}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "seed_sweep_" if recipe["train_batch_size"] == 12 else "seed_batch{}_".format(recipe["train_batch_size"])
    suite = (Path(output) if output else project / "runs/taxosieve" / (prefix + stamp + "_" + uuid.uuid4().hex[:8])).resolve()
    suite.mkdir(parents=True, exist_ok=False)
    runtime = suite / "runtime"
    for name in files:
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, destination)
        if digest(destination) != source_hashes[str(name)]:
            raise ValueError("Source changed while copying: " + str(name))
    (runtime / "prepro").symlink_to(project / "prepro", target_is_directory=True)
    patch_runtime(runtime)
    if recipe["train_batch_size"] != 12:
        ref["data"]["batch_size"] = recipe["train_batch_size"]
        ref["data"]["sampler"]["images_per_species"] = recipe["images_per_species"]
        ref["data"]["sampler"]["batches_per_epoch"] = recipe["batches_per_epoch"]
        (runtime / cfg["reference_config"]).write_text(
            yaml.safe_dump(ref, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), suite / "driver.py")
    snapshot_hashes = {str(name): digest(runtime / name) for name in files}
    for name in files:
        (runtime / name).chmod(0o444)
    (suite / "driver.py").chmod(0o444)
    plan = dict(schema="taxosieve_seed_sweep_v2", project_root=str(project), seeds=seeds,
        device=device, created_at_utc=stamp, runtime=str(runtime), source_hashes=source_hashes,
        snapshot_hashes=snapshot_hashes, source_inputs=input_hashes,
        reference_recipe=recipe,
        driver_sha256=digest(suite / "driver.py"), seed_scope="reference,D05,D05_calibration,staged_calibration_OOF",
        fixed_data_split=True, selection_rule=RANKING,
        phase_order="all TRAIN/DEV, freeze OOF selection, all frozen TEST",
        original_source_modified=False, original_seed1_modified=False)
    write_json(suite / "plan.json", plan)
    (suite / "plan.json").chmod(0o444)
    (suite / "logs").mkdir()
    verify_suite(suite)
    return suite


def verify_suite(suite):
    import yaml
    suite = Path(suite).resolve()
    plan = read_json(suite / "plan.json")
    if plan.get("schema") not in ("taxosieve_seed_sweep_v1", "taxosieve_seed_sweep_v2"):
        raise ValueError("Unknown sweep plan")
    if plan["schema"] == "taxosieve_seed_sweep_v2" and not isinstance(plan.get("reference_recipe"), dict):
        raise ValueError("Missing frozen reference recipe metadata")
    validate_seeds(plan["seeds"])
    runtime = suite / "runtime"
    if str(runtime) != plan["runtime"]:
        raise ValueError("The suite must remain at its original absolute path")
    if (runtime / "prepro").resolve() != (Path(plan["project_root"]) / "prepro").resolve():
        raise ValueError("Dataset link changed")
    present = {str(p.relative_to(runtime)) for package in PACKAGES
               for p in (runtime / package).glob("*.py")}
    present.update(str(p.relative_to(runtime)) for p in (runtime / "configs").glob("taxosieve*.yml"))
    present.update((*ROOT_FILES, "models/bpe_simple_vocab_16e6.txt.gz"))
    if present != set(plan["snapshot_hashes"]):
        raise ValueError("Frozen runtime file inventory changed")
    for name, expected in plan["snapshot_hashes"].items():
        if digest(runtime / name) != expected:
            raise ValueError("Frozen runtime changed: " + name)
    for name, expected in plan["source_inputs"].items():
        if digest(name) != expected:
            raise ValueError("Frozen data/config input changed: " + name)
    if digest(suite / "driver.py") != plan["driver_sha256"]:
        raise ValueError("Sweep driver changed")
    # Older seed-only plans have no batch metadata and remain readable.
    if "reference_recipe" in plan:
        source = yaml.safe_load((Path(plan["project_root"]) / "configs/taxosieve_reference.yml").read_text(encoding="utf-8"))
        expected = batch_recipe(source, plan["reference_recipe"]["train_batch_size"])
        if plan["reference_recipe"] != expected:
            raise ValueError("Reference recipe metadata changed")
        source["data"]["batch_size"] = expected["train_batch_size"]
        source["data"]["sampler"]["images_per_species"] = expected["images_per_species"]
        source["data"]["sampler"]["batches_per_epoch"] = expected["batches_per_epoch"]
        actual = yaml.safe_load((runtime / "configs/taxosieve_reference.yml").read_text(encoding="utf-8"))
        if source != actual:
            raise ValueError("Frozen reference configuration differs from the declared batch recipe")
    return plan


def metrics(summary):
    raw = (summary or {}).get("metrics", {})
    return {short: raw.get(key) for short, key in METRICS.items()}


def compare_audits(suite, include_test=True):
    """Compare actual decoded image identities from stage receipts across seeds."""
    suite = Path(suite)
    plan = verify_suite(suite)
    baseline = {}
    receipts = ("reference/training/completed.json", "reference/calibration/completed.json")
    if include_test:
        receipts += ("cache/test/completed.json",)
    for seed in plan["seeds"]:
        for relative in receipts:
            path = suite / ("seed_" + str(seed)) / relative
            if not path.is_file():
                continue
            audit = read_json(path)["audit"]
            for split, entry in audit.items():
                identity = dict(image_hashes=sorted(entry["image_hashes"]),
                    count=entry["count"], unique_image_count=entry["unique_image_count"],
                    manifest_sha256=entry["manifest_sha256"], sources=sorted(entry["sources"]))
                if split in baseline and baseline[split][1] != identity:
                    raise ValueError("Actual image data differs across seeds for {}: seed {} vs seed {}. Preserve the suite; do not compare these runs.".format(split, baseline[split][0], seed))
                baseline.setdefault(split, (seed, identity))
    return {split: entry for split, (_, entry) in baseline.items()}


def select_seed(suite):
    """Use only completed DEV OOF; no TEST file is read by this function."""
    suite = Path(suite)
    plan = verify_suite(suite)
    compare_audits(suite, include_test=False)
    candidates = []
    for seed in plan["seeds"]:
        audit = read_json(suite / ("seed_" + str(seed)) / "calibration/audit.json")
        oof = audit["crossfit"]
        report = oof.get("report") or {}
        values = metrics(report)
        eligible = oof.get("complete") is True and all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values.values())
        candidates.append(dict(seed=seed, eligible=eligible, oof_complete=oof.get("complete"),
            crossfit_passed=oof.get("passed"), targets_passed=report.get("targets_passed"), metrics=values))
    def ranking(row):
        m = row["metrics"]
        return (row["targets_passed"] is True, min(m.values()), m["near"], m["known"],
                m["extra"], m["precision"], -row["seed"])
    eligible = [r for r in candidates if r["eligible"]]
    selected = max(eligible, key=ranking)["seed"] if eligible else None
    result = dict(schema="taxosieve_seed_selection_v1", selection_rule=RANKING,
        recommended_seed=selected, candidates=candidates, test_used_for_selection=False,
        recommended_targets_passed=next((r["targets_passed"] for r in candidates if r["seed"] == selected), None),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        interpretation="Exploratory conditional OOF comparison; not independent model-level validation")
    write_json(suite / "selection.json", result)
    (suite / "selection.json").chmod(0o444)
    return result


def summarize_suite(suite):
    suite = Path(suite)
    plan = verify_suite(suite)
    audit = compare_audits(suite)
    recipe = plan.get("reference_recipe", {})
    rows = []
    for seed in plan["seeds"]:
        run = suite / ("seed_" + str(seed))
        fit = read_json(run / "calibration/completed.json")["summary"]
        oof = read_json(run / "calibration/audit.json")["crossfit"]
        test = read_json(run / "test/completed.json")["summary"]
        for phase, summary, complete in (("DEV-fit", fit, True), ("DEV-OOF", oof.get("report"), oof.get("complete")), ("TEST", test, True)):
            rows.append(dict(seed=seed, phase=phase, complete=complete,
                train_batch_size=recipe.get("train_batch_size", 12),
                reference_batches_per_epoch=recipe.get("batches_per_epoch", 240),
                targets_passed=(summary or {}).get("targets_passed"), **metrics(summary)))
    tests = [r for r in rows if r["phase"] == "TEST"]
    aggregates = {}
    for key in METRICS:
        values = [r[key] for r in tests if type(r[key]) in (int, float) and math.isfinite(r[key])]
        aggregates[key] = dict(n_valid=len(values), mean=statistics.mean(values) if values else None,
            sample_std=statistics.stdev(values) if len(values) > 1 else None)
    result = dict(rows=rows, test_statistics=aggregates, selection=read_json(suite / "selection.json"),
        reference_recipe=recipe,
        metric_scale="fraction [0,1]", seeds=plan["seeds"], common_split_audit=audit)
    write_json(suite / "comparison.json", result)
    with (suite / "comparison.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "phase", "train_batch_size", "reference_batches_per_epoch", "complete", "targets_passed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    return result


def execute(suite, label, arguments, seed):
    plan = verify_suite(suite)
    runtime = Path(plan["runtime"])
    log = Path(suite) / "logs" / (label + ".log")
    command = [sys.executable, "-u", str(runtime / "run_taxosieve.py"), *map(str, arguments)]
    print("\n[{}] {}\nLog: {}".format(label, " ".join(command), log), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(seed))
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=str(runtime), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError("{} failed (exit {}). See {}. This suite is not restarted automatically; preserve it and use a fresh suite after fixing the cause.".format(label, code, log))
    verify_suite(suite)
    compare_audits(suite)


def run_suite(suite):
    suite = Path(suite).resolve()
    plan = verify_suite(suite)
    if any((suite / ("seed_" + str(seed))).exists() for seed in plan["seeds"]):
        raise ValueError("Seed run directories must be absent; never overwrite or automatically resume a suite")
    write_json(suite / "started.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat()))
    for seed in plan["seeds"]:
        run = suite / ("seed_" + str(seed))
        execute(suite, "seed_{}_preflight".format(seed), ["preflight", "--seed", seed], seed)
        execute(suite, "seed_{}_train".format(seed), ["train", "--run-dir", run, "--device", plan["device"], "--seed", seed], seed)
        execute(suite, "seed_{}_calibrate".format(seed), ["calibrate", "--run-dir", run, "--save-scores"], seed)
    selection = select_seed(suite)
    frozen = digest(suite / "selection.json")
    print("\nFrozen DEV-OOF recommended seed:", selection["recommended_seed"], flush=True)
    for seed in plan["seeds"]:
        if digest(suite / "selection.json") != frozen:
            raise ValueError("Frozen seed selection changed")
        execute(suite, "seed_{}_test".format(seed), ["test", "--run-dir", suite / ("seed_" + str(seed)), "--save-scores"], seed)
    if digest(suite / "selection.json") != frozen:
        raise ValueError("Frozen seed selection changed")
    result = summarize_suite(suite)
    print("\nFinished. Comparison:", suite / "comparison.csv", flush=True)
    def pct(value):
        return "N/A" if value is None else "{:.2%}".format(value)
    print("seed    Known       Near        Extra       Leaf PPV", flush=True)
    for row in result["rows"]:
        if row["phase"] == "TEST":
            print("{:<8}".format(row["seed"]) + "  ".join(
                "{:<10}".format(pct(row[key])) for key in METRICS), flush=True)
    for title, field in (("mean", "mean"), ("std", "sample_std")):
        print("{:<8}".format(title) + "  ".join("{:<10}".format(
            pct(result["test_statistics"][key][field])) for key in METRICS), flush=True)
    print("DEV-OOF recommended seed: {}; four metric targets passed: {}".format(
        result["selection"]["recommended_seed"], result["selection"]["recommended_targets_passed"]), flush=True)
    print("Full values, sample standard deviations and n_valid:", suite / "comparison.json", flush=True)
    return result


def find_project():
    for path in (Path.cwd(), *Path(__file__).resolve().parents):
        if (path / "run_taxosieve.py").is_file():
            return path
    raise ValueError("Cannot locate run_taxosieve.py; provide --project-root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", help="Default: 2 3 4 5")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Default: cuda")
    parser.add_argument("--train-batch-size", type=int,
        help="Image batch (default 12); adjusts images/species and batches/epoch to preserve 2880 sampled images/epoch; changes optimizer step count")
    parser.add_argument("--output", type=Path, help="An absent destination for the new suite")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--suite", type=Path, help="Start a previously prepared, never-started suite")
    args = parser.parse_args()
    if args.suite:
        if (args.prepare_only or args.project_root or args.output or args.seeds is not None
                or args.device is not None or args.train_batch_size is not None):
            parser.error("--suite uses its frozen plan and cannot be combined with preparation overrides")
        suite = args.suite.resolve()
        verify_suite(suite)
    else:
        suite = prepare_suite(args.project_root or find_project(), args.seeds or [2, 3, 4, 5],
            output=args.output, device=args.device or "cuda", train_batch_size=args.train_batch_size)
    print("Suite:", suite, flush=True)
    recipe = verify_suite(suite).get("reference_recipe")
    if recipe:
        print("Reference image recipe: " + json.dumps(recipe, ensure_ascii=False), flush=True)
        if not recipe["same_optimizer_steps_as_baseline"]:
            print("Batch-size ablation: same sampled images/epoch, fewer optimizer steps/epoch; compare separately from the original batch-12 sweep.", flush=True)
    if args.prepare_only:
        import shlex
        print("Prepared only; start once with:\n" + " ".join(shlex.quote(str(x)) for x in
            (sys.executable, suite / "driver.py", "--suite", suite)), flush=True)
    else:
        run_suite(suite)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit("ERROR: " + str(error))
