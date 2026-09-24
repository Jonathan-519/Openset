"""Bundle one run's receipts, scores and metrics, excluding model/image bytes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def pack(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Review archive already exists: " + str(output))
    cfg = json.loads((run / "training/config.json").read_text(encoding="utf-8"))
    files = [(p, "run/" + p.relative_to(run).as_posix()) for p in sorted(run.rglob("*"))
             if p.is_file() and p.suffix in {".json", ".jsonl", ".log", ".md", ".txt", ".yml"}]
    for split in ("train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra", "known_preparation_audit"):
        if not cfg["data"].get(split):
            continue
        path = Path(cfg["data"][split])
        path = path if path.is_absolute() else ROOT / path
        if path.is_file():
            files.append((path, "manifests/" + split + path.suffix))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
    print("Review archive: " + str(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/taxosafe_v11_dcbs/main/trial_1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir.with_name(args.run_dir.name + "_review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".tar.gz")
    pack(args.run_dir, output)
