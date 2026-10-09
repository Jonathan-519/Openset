"""Plan, train and evaluate the TaxoSafe-HLE development hypothesis.

Run --stage develop does NOT read test images. Test is a separate command.
Completed-stage resume verifies hashes; partial stages require a new suite.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from taxosafe_visual.runtime import resolve, sha256, read_json, write_json

CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_hier_evidence_v4.yml"
STAGES = ("cache_train", "train", "cache_val", "calibrate", "cache_test", "test")


def portable(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def make_plan(source_suite, suite_dir, seed, config=CONFIG):
    from tools.plan_taxosafe_partial_pooling import verified_training_source
    from tools import run_taxosafe_suite as old_suite
    original, source_plan, train_receipt, archive = verified_training_source(source_suite, seed)
    for stage in ("memory", "calibrate"):
        item = [s for s in old_suite.steps(original, stage) if s["id"] == "full_" + stage][0]
        if not old_suite.verify_receipt(original, item):
            raise ValueError("Original full/{} has no verified receipt".format(stage))
    artifacts = resolve(original["variants"]["full"]["artifacts"])
    bank = read_json(artifacts / "memory.json")
    cal = read_json(artifacts / "calibration.json")
    checkpoint = resolve(original["run_dir"]) / "ckpt/best.pth"
    if bank["checkpoint_sha256"] != sha256(checkpoint) or cal["metadata"]["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Original root checkpoint provenance mismatch")
    settings = yaml.safe_load(resolve(config).read_text(encoding="utf-8-sig"))
    from taxosafe_hier.model import VARIANTS
    if settings["variants"] != list(VARIANTS) or settings["primary_variant"] != "full" or settings["primary_profile"] != "protected":
        raise ValueError("This study must report all six preregistered variants and full/protected")
    cfg = yaml.safe_load(archive.read_text(encoding="utf-8-sig"))
    inputs = {source_plan, train_receipt, archive, checkpoint, resolve(config),
              artifacts / "memory.npz", artifacts / "memory.json", artifacts / "calibration.json"}
    inputs.update(resolve(cfg["data"][key]) for key in (
        "train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra", "hierarchy"))
    for directory in ("models", "loader", "taxosafe_visual", "taxosafe_hier", "tools"):
        inputs.update((ROOT / directory).rglob("*.py"))
    bpe = ROOT / "models/bpe_simple_vocab_16e6.txt.gz"
    if not bpe.is_file():
        raise FileNotFoundError("Install the verified CLIP BPE vocabulary before planning")
    inputs.add(bpe)
    fingerprint = {portable(path): sha256(path) for path in sorted(inputs)}
    folder = resolve(suite_dir) / ("seed_" + str(seed))
    plan = {"schema": 1, "seed": seed, "suite": portable(folder), "source_plan": portable(source_plan),
            "checkpoint": portable(checkpoint), "training_config": portable(archive),
            "root_memory": portable(artifacts / "memory.npz"), "root_calibration": portable(artifacts / "calibration.json"),
            "taxonomy": bank["taxonomy"], "settings": settings, "inputs_sha256": fingerprint,
            "protocol": "Development only. Original test has influenced prior method design. Fresh outer species/acquisition split required."}
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", plan)
    return folder / "plan.json"


def outputs(plan, stage):
    variants = plan["settings"]["variants"]
    if stage.startswith("cache_"):
        name = stage[6:]
        return ["cache/" + name + suffix for suffix in (".npz", ".jsonl")]
    if stage == "train":
        return ["artifacts/{}/{}".format(v, f) for v in variants for f in ("adapter.npz", "training.json")]
    if stage == "calibrate":
        return ["validation_report.json"] + ["artifacts/" + v + "/calibration.json" for v in variants]
    if stage == "test":
        return ["summary.json"] + ["artifacts/{}/test/{}/{}".format(v, p, f) for v in variants
                                  for p in ("coverage", "balanced", "protected") for f in ("metrics.json", "predictions.jsonl")]
    raise ValueError("Unknown stage")


def verify_inputs(plan):
    for path, expected in plan["inputs_sha256"].items():
        if not resolve(path).is_file() or sha256(resolve(path)) != expected:
            raise ValueError("Frozen input changed: {}. Create a new suite; do not rewrite hashes.".format(path))


def verified(plan, stage):
    folder = resolve(plan["suite"])
    receipt = folder / "receipts" / (stage + ".json")
    if not receipt.exists():
        return False
    info = read_json(receipt)
    if info["plan_sha256"] != sha256(folder / "plan.json") or set(info["outputs"]) != set(outputs(plan, stage)):
        raise ValueError("Receipt/plan mismatch: " + stage)
    for path, expected in info["outputs"].items():
        if not (folder / path).is_file() or sha256(folder / path) != expected:
            raise ValueError("Completed output changed: " + path)
    return True


def run(plan, selection, device, resume=False, dry_run=False):
    folder = resolve(plan["suite"])
    verify_inputs(plan)
    selected = STAGES[:4] if selection == "develop" else STAGES if selection == "all" else (
        STAGES[4:] if selection == "test" else (selection,))
    for stage in selected:
        earlier = STAGES[:STAGES.index(stage)]
        if resume and verified(plan, stage):
            if not all(verified(plan, s) for s in earlier):
                raise ValueError("Missing prerequisite for completed stage: " + stage)
            print("Verified complete:", stage)
            continue
        if any((folder / path).exists() for path in outputs(plan, stage)):
            raise FileExistsError("Existing/partial outputs at {}. Use --resume for completed steps; retain failed suite and use a new suite for partial steps.".format(stage))
        argv = [sys.executable, "-u", str(Path(__file__).resolve()), "_step", "--plan", str(folder / "plan.json"), "--stage", stage, "--device", device]
        if dry_run:
            import shlex
            print(shlex.join(argv))
            continue
        if not all(verified(plan, s) for s in earlier):
            raise ValueError("Incomplete prerequisite before " + stage)
        log = folder / "logs" / (stage + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            with subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True); stream.write(line)
                status = process.wait()
        if status:
            raise RuntimeError("{} failed, see {}".format(stage, log))
        write_json(folder / "receipts" / (stage + ".json"), {
            "plan_sha256": sha256(folder / "plan.json"),
            "outputs": {path: sha256(folder / path) for path in outputs(plan, stage)}})


def summarize(suite_dir, output_dir):
    import statistics
    source, dest = resolve(suite_dir).resolve(), resolve(output_dir).resolve()
    if dest == source or source in dest.parents or dest.exists():
        raise ValueError("Choose a new diagnosis directory outside the suite")
    rows = []
    for seed in (1, 2, 3):
        plan = read_json(source / ("seed_" + str(seed)) / "plan.json")
        verify_inputs(plan)
        if not all(verified(plan, s) for s in STAGES):
            raise ValueError("Three fully completed seeds are required")
        result = read_json(resolve(plan["suite"]) / "summary.json")
        rows.extend({"seed": seed, **row} for row in result["rows"])
    fields = ("closed_routed_accuracy", "closed_routed_macro_accuracy", "known_end_to_end_leaf_accuracy",
              "intra_cfr", "intra_oser", "extra_far", "intra_macro_parent_species_auroc", "drta")
    table = ["Development results: mean +/- sample SD (%); NOT confirmed SOTA.", "",
             "|method|profile|" + "|".join(fields) + "|", "|---|---|" + "---:|" * len(fields)]
    for method, profile in sorted({(r["method"], r["profile"]) for r in rows}):
        group = [r for r in rows if r["method"] == method and r["profile"] == profile]
        cells = []
        for field in fields:
            vals = [100 * r[field] for r in group if r[field] is not None]
            cells.append("{:.2f} +/- {:.2f}".format(statistics.mean(vals), statistics.stdev(vals)) if len(vals) == 3 else "NA")
        table.append("|" + "|".join([method, profile] + cells) + "|")
    dest.mkdir(parents=True)
    write_json(dest / "summary.json", {"rows": rows})
    (dest / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("Summary:", dest / "summary.md")


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--source-suite", default="runs/taxosafe_rs_paper")
    p.add_argument("--suite-dir", default="runs/taxosafe_hier_v4_dev")
    p.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--config", default=CONFIG)
    for command in ("run", "_step"):
        p = sub.add_parser(command)
        p.add_argument("--plan", required=True)
        p.add_argument("--stage", choices=STAGES + ("develop", "all"), default="develop")
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--suite-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print("Frozen plan:", make_plan(args.source_suite, args.suite_dir, args.seed, args.config))
    elif args.command == "summarize":
        summarize(args.suite_dir, args.output_dir)
    else:
        plan = read_json(resolve(args.plan))
        if args.command == "run":
            run(plan, args.stage, args.device, args.resume, args.dry_run)
        else:
            from taxosafe_hier import pipeline
            verify_inputs(plan)
            if args.stage.startswith("cache_"):
                pipeline.cache(plan, args.stage[6:])
            elif args.stage in ("train", "calibrate", "test"):
                getattr(pipeline, args.stage)(plan, args.device)
            else:
                raise ValueError("_step requires a single concrete stage")


if __name__ == "__main__":
    main()
