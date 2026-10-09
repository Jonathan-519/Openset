"""Bundle a sweep's text diagnostics, including incomplete and failed arms.

The source suite is read only. Images, models, feature caches and links are
never copied; the resulting archive is a review bundle, not a model backup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile


SUITE_SCHEMA_VERSION = "taxosafe_domain_v1"
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml", ".csv",
    ".tsv", ".stdout", ".stderr", ".out", ".err",
}
TEXT_NAMES = {"stdout", "stderr"}
EXCLUDED_DIRECTORIES = {
    "raw", "images", "image", "checkpoints", "checkpoint", "models",
    "weights", "cache", "caches", "feature_cache", "feature_caches",
    "features", ".git", "__pycache__", ".pytest_cache",
}


def _open_directory(path, create=False):
    """Open each path component without following even ancestor symlinks."""
    path = Path(path).absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ValueError("Directory must be real and have no symlink ancestors: " + str(path)) from error


def _read_regular(directory, name):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Expected a regular file: " + name)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            return handle.read()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _excluded_directory(name):
    lowered = name.lower()
    return (lowered in EXCLUDED_DIRECTORIES or lowered.endswith("_cache")
            or lowered.endswith("_caches"))


def _text_files(directory, prefix=Path()):
    """Traverse directory descriptors so renamed/replaced links are not followed."""
    for name in sorted(os.listdir(directory)):
        metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        relative = prefix / name
        if stat.S_ISDIR(metadata.st_mode):
            if _excluded_directory(name) and relative not in (Path("cache"), Path("logs/cache")):
                continue
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=directory)
            try:
                yield from _text_files(child, relative)
            finally:
                os.close(child)
        elif (stat.S_ISREG(metadata.st_mode)
              and (relative.suffix.lower() in TEXT_SUFFIXES or name.lower() in TEXT_NAMES)):
            if relative.parts[0] == "cache" and name not in {"completed.json", "failure.json", "stage_binding.json"}:
                continue
            yield relative, _read_regular(directory, name)


def _add_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(payload), 0o644
    archive.addfile(info, io.BytesIO(payload))


def pack(suite, output=None):
    suite = Path(suite).absolute()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(output).absolute() if output is not None else
              suite.with_name(suite.name + "_review_" + stamp + ".tar.gz"))
    # Normalize only for containment checking; actual I/O still checks every
    # originally supplied component, including paths containing '..'.
    normalized_suite = Path(os.path.abspath(str(suite)))
    normalized_output = Path(os.path.abspath(str(output)))
    if normalized_output == normalized_suite or normalized_suite in normalized_output.parents:
        raise ValueError("Archive destination must be outside the read-only source suite")

    source = _open_directory(suite)
    destination_parent = None
    created = False
    try:
        try:
            snapshot_payload = _read_regular(source, "snapshot.json")
            snapshot = json.loads(snapshot_payload.decode("utf-8"))
        except (OSError, ValueError, UnicodeError) as error:
            raise ValueError("Expected a regular, valid suite snapshot.json") from error
        if (not isinstance(snapshot, dict)
                or snapshot.get("schema_version") != SUITE_SCHEMA_VERSION):
            raise ValueError("Expected snapshot.json schema_version=" + SUITE_SCHEMA_VERSION)

        destination_parent = _open_directory(output.parent, create=True)
        try:
            descriptor = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o644, dir_fd=destination_parent)
        except OSError as error:
            raise ValueError("Archive destination must be new and must not be a symlink: " + str(output)) from error
        created = True
        manifest = {
            "schema_version": "taxosafe_domain_review_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "suite_name": suite.name,
            "suite_schema_version": SUITE_SCHEMA_VERSION,
            "includes_images": False,
            "includes_checkpoints": False,
            "includes_feature_caches": False,
            "includes_external_data_manifests": False,
            "is_model_backup": False,
            "files": [],
            "skipped_non_text_files": [],
        }
        with os.fdopen(descriptor, "wb") as handle:
            with tarfile.open(fileobj=handle, mode="w:gz") as archive:
                for relative, payload in _text_files(source):
                    name = "suite/" + relative.as_posix()
                    if relative == Path("snapshot.json"):
                        payload = snapshot_payload
                    try:
                        payload.decode("utf-8")
                        if b"\0" in payload:
                            raise ValueError("NUL byte")
                    except (UnicodeError, ValueError):
                        manifest["skipped_non_text_files"].append(name)
                        continue
                    _add_bytes(archive, name, payload)
                    manifest["files"].append({
                        "path": name, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    })
                payload = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n").encode("utf-8")
                _add_bytes(archive, "archive_manifest.json", payload)
    except BaseException:
        if created:
            os.unlink(output.name, dir_fd=destination_parent)
        raise
    finally:
        os.close(source)
        if destination_parent is not None:
            os.close(destination_parent)
    print("Review archive: " + str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pack(args.suite_dir, args.output)


if __name__ == "__main__":
    main()
