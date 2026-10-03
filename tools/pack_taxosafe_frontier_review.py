"""Package Frontier text diagnostics; never follow source links or copy models."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".csv"}
EXCLUDED_DIRECTORIES = {"raw", "images", "checkpoints", ".git", "__pycache__"}


def _no_symlink(path):
    return not any(item.is_symlink() for item in (path, *path.parents))


def pack(run, output):
    run, output = Path(run).absolute(), Path(output).absolute()
    if not _no_symlink(run) or not run.is_dir():
        raise ValueError("Run must be a real directory, with no symlink ancestors")
    if not _no_symlink(output) or output.exists():
        raise ValueError("Archive destination must be new and must not traverse symlinks")
    config = run / "frontier/config.json"
    if not _no_symlink(config) or not config.is_file():
        raise ValueError("Expected frontier/config.json in the explicit run directory")
    configuration = json.loads(config.read_text(encoding="utf-8"))
    if (not isinstance(configuration.get("geometry"), dict)
            or not isinstance(configuration.get("calibration"), dict)
            or "support" in configuration or "dcbs" in configuration):
        raise ValueError("Expected a frozen Frontier configuration")
    files = []
    for directory, children, names in os.walk(str(run), followlinks=False):
        root = Path(directory)
        children[:] = sorted(name for name in children
                             if name not in EXCLUDED_DIRECTORIES
                             and not (root / name).is_symlink())
        for name in sorted(names):
            path = root / name
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path == output:
                continue
            files.append((path, "run/" + path.relative_to(run).as_posix()))
    files.sort(key=lambda item: item[1])
    if not files:
        raise ValueError("No text diagnostics found")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "taxosafe_frontier_review_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run.name,
        "includes_images": False, "includes_checkpoints": False,
        "includes_feature_caches": False, "includes_external_data_manifests": False,
        "is_model_backup": False,
        "files": [],
    }
    with tarfile.open(str(output), "x:gz") as archive:
        for path, name in files:
            payload = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))
            metadata["files"].append({"path": name, "bytes": len(payload),
                                      "sha256": hashlib.sha256(payload).hexdigest()})
        payload = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo("archive_manifest.json")
        info.size, info.mode = len(payload), 0o644
        archive.addfile(info, io.BytesIO(payload))
    print("Review archive: " + str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.run_dir.with_name(args.run_dir.name + "_review_" + stamp + ".tar.gz")
    pack(args.run_dir, output)


if __name__ == "__main__":
    main()
