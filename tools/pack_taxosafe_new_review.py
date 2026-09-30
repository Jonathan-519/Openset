"""Package one TaxoSafe-new run's text diagnostics, without weights or images."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yml", ".yaml", ".csv"}
EXCLUDED_DIRECTORIES = {"raw", "images", "checkpoints", ".git", "__pycache__"}


def pack(run, output):
    """Write an exclusive archive; permit failed runs with a saved config.

    Only regular diagnostic files below ``run`` are read. Symlinks, including
    symlinked parent directories, are never followed. Dataset paths mentioned
    in config files are not opened or copied.
    """
    run, output = Path(run).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Review archive already exists: " + str(output))
    config = run / "training/config.json"
    if not config.is_file() or config.is_symlink() or config.parent.is_symlink():
        raise ValueError("Expected an existing run with training/config.json: " + str(run))
    configuration = json.loads(config.read_text(encoding="utf-8"))
    if "support" not in configuration or "dcbs" in configuration:
        raise ValueError("Expected a TaxoSafe-new support configuration, not a historical run")
    files = []
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p != run and run in p.parents):
            continue
        files.append((path, "run/" + relative.as_posix()))
    if not files:
        raise ValueError("No text diagnostics found: " + str(run))

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "taxosafe_new_review_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run.name,
        "includes_images": False,
        "includes_checkpoints": False,
        "includes_external_data_manifests": False,
        "files": [],
    }
    # Read each member once: its checksum describes exactly the archived bytes,
    # even if a caller intentionally packages diagnostics while a job is active.
    with tarfile.open(output, "x:gz") as archive:
        for path, name in files:
            payload = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
            metadata["files"].append({"path": name, "bytes": len(payload),
                                      "sha256": hashlib.sha256(payload).hexdigest()})
        payload = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo("archive_manifest.json")
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    print("Review archive: " + str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "runs/taxosafe_new/main/trial_1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.run_dir.with_name(args.run_dir.name + "_review_" + stamp + ".tar.gz")
    pack(args.run_dir, output)


if __name__ == "__main__":
    main()
