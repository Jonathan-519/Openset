#!/usr/bin/env python3
"""Refresh H02 manifest filenames without changing image identities or splits.

Python 3.8+ standard library only. The default is a read-only preview; --apply
backs up existing outputs before replacing them. Images, configurations,
taxonomy and completed runs are never modified. A changed manifest requires a
new run directory because previous runs bind the original manifest bytes.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile


BASELINE_SHA256 = "255e276b0c7aab69f6ee8314b5487ea3d3c3832d0c9df4d81ac6f46d6bdc7726"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "reproducibility/history/filename_baseline.json"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _relative(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\\\r\n\0"):
        raise ValueError("Invalid relative path: " + repr(value))
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or path.as_posix() != value:
        raise ValueError("Path must be canonical and project-relative: " + value)
    return path


def _project_root(value):
    root = Path(value).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Project root must be an existing non-symlink directory")
    return root.resolve()


def _safe_path(root, relative):
    path = root
    for part in _relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed: " + relative)
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escapes the project: " + relative) from exc
    return path


def _read_optional(root, relative):
    path = _safe_path(root, relative)
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("Expected a regular manifest/audit file: " + relative)
    return path.read_bytes().decode("utf-8")


def _validate_baseline(baseline):
    if baseline.get("schema_version") != "h02_filename_baseline_v1":
        raise ValueError("Unsupported H02 filename baseline schema")
    if not baseline.get("raw_files") or not baseline.get("manifests"):
        raise ValueError("Baseline image and manifest inventories must be nonempty")
    seen = set()
    for row in baseline["raw_files"]:
        _relative(row["path"])
        if row["path"] in seen or PurePosixPath(row["path"]).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Invalid or duplicate baseline image: " + row["path"])
        if (type(row["size"]) is not int or row["size"] < 0 or len(row["blob_sha1"]) != 40
                or any(c not in "0123456789abcdef" for c in row["blob_sha1"])):
            raise ValueError("Invalid baseline image identity: " + row["path"])
        seen.add(row["path"])
    for path, entry in baseline["manifests"].items():
        _relative(path)
        _relative(entry["root"])
        if _sha256(entry["content"].encode("utf-8")) != entry["sha256"]:
            raise ValueError("Baseline manifest content hash differs: " + path)
    audit = baseline["known_audit"]
    _relative(audit["path"])
    if audit["path"] in baseline["manifests"] or _sha256(audit["content"].encode("utf-8")) != audit["sha256"]:
        raise ValueError("Baseline known audit content hash or path differs")
    if set(baseline["protected_files"]) & (set(baseline["manifests"]) | {audit["path"]}):
        raise ValueError("A protected file cannot also be a refresh output")


def load_baseline(path=DEFAULT_BASELINE):
    """Load only the exact reviewed baseline, never an unchecked replacement."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Baseline must be a regular file")
    data = path.read_bytes()
    if _sha256(data) != BASELINE_SHA256:
        raise ValueError("Baseline SHA256 differs from the pinned H02 baseline")
    baseline = json.loads(data.decode("utf-8"))
    _validate_baseline(baseline)
    return baseline


def _stat_token(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _hash_image(path, relative):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Image must be a regular file: " + relative)
    blob = hashlib.sha1(b"blob " + str(before.st_size).encode("ascii") + b"\0")
    sha = hashlib.sha256()
    count = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(str(path), flags), "rb") as handle:
        if _stat_token(os.fstat(handle.fileno())) != _stat_token(before):
            raise ValueError("Image changed before hashing: " + relative)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += len(block)
            blob.update(block)
            sha.update(block)
        after = os.fstat(handle.fileno())
    if (count != before.st_size or _stat_token(before) != _stat_token(after)
            or _stat_token(before) != _stat_token(path.lstat())):
        raise ValueError("Image changed while hashing: " + relative)
    return dict(path=relative, blob_sha1=blob.hexdigest(), size=count,
                sha256=sha.hexdigest(), stat=_stat_token(before))


def scan_images(project_root, baseline):
    """Hash each image once, strictly within the baseline role directories."""
    root = _project_root(project_root)
    roles = sorted({_relative(entry["root"]).as_posix() for entry in baseline["manifests"].values()})
    roots = [role for role in roles if not any(role.startswith(other + "/") for other in roles)]
    allowed_dirs = set(roots)
    for row in baseline["raw_files"]:
        path = _relative(row["path"])
        containing = [role for role in roots if row["path"].startswith(role + "/")]
        if len(containing) != 1:
            raise ValueError("Baseline image is outside its role roots: " + row["path"])
        parent = path.parent
        while parent.as_posix() != containing[0]:
            allowed_dirs.add(parent.as_posix())
            parent = parent.parent
    result = []

    def walk(relative):
        directory = _safe_path(root, relative)
        if not directory.is_dir():
            raise ValueError("Missing image directory: " + relative)
        with os.scandir(str(directory)) as entries:
            items = sorted(entries, key=lambda entry: entry.name)
        for entry in items:
            child = relative + "/" + entry.name
            _relative(child)
            if entry.is_symlink():
                raise ValueError("Symlinks are not allowed in image roots: " + child)
            if entry.is_dir(follow_symlinks=False):
                if child not in allowed_dirs:
                    raise ValueError("New or moved image directory: " + child)
                walk(child)
            elif PurePosixPath(child).suffix.lower() in IMAGE_EXTENSIONS:
                result.append(_hash_image(Path(entry.path), child))

    for role in roots:
        walk(role)
    return sorted(result, key=lambda row: row["path"])


def build_mapping(baseline_files, current_files):
    """Match same-directory content multisets, retaining unchanged aliases first."""
    groups = []
    for rows in (baseline_files, current_files):
        values = defaultdict(list)
        seen = set()
        for row in rows:
            path = _relative(row["path"])
            if row["path"] in seen:
                raise ValueError("Duplicate image inventory path: " + row["path"])
            seen.add(row["path"])
            values[(path.parent.as_posix(), row["blob_sha1"], row["size"])].append(row["path"])
        groups.append(values)
    old, new = groups
    old_counts = Counter({key: len(value) for key, value in old.items()})
    new_counts = Counter({key: len(value) for key, value in new.items()})
    if old_counts != new_counts:
        missing, extra = old_counts - new_counts, new_counts - old_counts
        examples = ["missing/changed: " + path for key in sorted(missing) for path in sorted(old[key])[:1]]
        examples += ["extra/changed: " + path for key in sorted(extra) for path in sorted(new[key])[:1]]
        raise ValueError("Image contents/counts changed or images moved between directories "
                         "(missing=%d, extra=%d): %s" % (sum(missing.values()), sum(extra.values()), "; ".join(examples[:8])))
    mapping = {}
    for key in sorted(old):
        shared = set(old[key]) & set(new[key])
        mapping.update((path, path) for path in sorted(shared))
        mapping.update(zip(sorted(set(old[key]) - shared), sorted(set(new[key]) - shared)))
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Image filename mapping is not bijective")
    return mapping


def _manifest_rows(content):
    for line in content.splitlines(keepends=True):
        if not line.strip():
            yield None, line
            continue
        body = line.rstrip("\r\n")
        try:
            path, label, index = body.rsplit(",", 2)
            int(label)
            int(index)
        except ValueError as exc:
            raise ValueError("Invalid manifest path,label,index row: " + repr(body)) from exc
        _relative(path)
        yield (path, label, index), line


def render_manifests(baseline, mapping):
    """Replace only path columns; retain row order, label/index tokens and EOLs."""
    outputs = {}
    for name, entry in baseline["manifests"].items():
        role = _relative(entry["root"])
        lines = []
        for fields, original in _manifest_rows(entry["content"]):
            if fields is None:
                lines.append(original)
                continue
            old_path = (role / fields[0]).as_posix()
            if old_path not in mapping:
                raise ValueError("Manifest image has no verified mapping: " + old_path)
            mapped = _relative(mapping[old_path]).relative_to(role).as_posix()
            lines.append(mapped + original[len(fields[0]):])
        outputs[name] = "".join(lines)
    return outputs


def _known_audit(baseline, outputs, mapping, current_files):
    """Independently reproduce prepare(): TEST priority, then VAL, then TRAIN."""
    original = json.loads(baseline["known_audit"]["content"])
    current = {row["path"]: row["sha256"] for row in current_files}
    rows, inputs, identities, roots = {}, {}, {}, {}
    for split in ("train", "val_known", "test_known"):
        manifest = original["inputs"][split]["manifest"]
        entry = baseline["manifests"][manifest]
        if original["inputs"][split]["sha256"] != entry["sha256"]:
            raise ValueError("Baseline audit and input manifest disagree: " + split)
        roots[split] = _relative(entry["root"])
        inputs[split] = dict(manifest=manifest, sha256=_sha256(outputs[manifest].encode("utf-8")))
        rows[split] = []
        for fields, line in _manifest_rows(outputs[manifest]):
            if fields is None:
                continue
            path, label, _ = fields
            label = int(label)
            sha = current[(roots[split] / path).as_posix()]
            if sha in identities and identities[sha] != label:
                raise ValueError("Same image bytes carry conflicting known labels: " + path)
            identities[sha] = label
            rows[split].append(dict(path=path, label=label, sha256=sha, line=line.rstrip("\r\n")))
    owners = {r["sha256"]: ("test_known", r["path"]) for r in rows["test_known"]}
    kept, removed = {"test_known": rows["test_known"]}, []
    for split in ("val_known", "train"):
        kept[split] = []
        for row in rows[split]:
            sha = row["sha256"]
            if sha in owners:
                other_split, other_path = owners[sha]
                removed.append(dict(split=split, path=row["path"], sha256=sha,
                                    same_as_split=other_split, same_as_path=other_path))
            else:
                kept[split].append(row)
                owners[sha] = split, row["path"]
    if any(not kept[split] for split in kept):
        raise ValueError("Identity preparation would empty a split")
    if {r["label"] for r in kept["train"]} != {r["label"] for r in rows["train"]}:
        raise ValueError("Identity preparation would remove all training support for a known leaf")
    report = dict(schema_version=11, operation="known_image_identity_only", unknown_images_read=False,
        test_known_used_only_for_identity_audit=True, locked_test_manifest_changed=False, inputs=inputs,
        original_counts={s: len(r) for s, r in rows.items()},
        retained_counts={s: len(kept[s]) for s in rows},
        unique_test_known_images=len({r["sha256"] for r in rows["test_known"]}),
        validation_labels_without_independent_images=sorted(
            {r["label"] for r in rows["val_known"]} - {r["label"] for r in kept["val_known"]}),
        removed_fitting_rows=removed)
    parent = _relative(baseline["known_audit"]["path"]).parent
    for name, split in (("gt_train_known.txt", "train"), ("gt_val_known.txt", "val_known")):
        path = (parent / name).as_posix()
        recomputed = "\n".join(row["line"] for row in kept[split]) + "\n"
        if outputs.get(path) != recomputed:
            raise ValueError("Rebuilt v11 split differs from the mapped locked baseline: " + path)
    inverse = {new: old for old, new in mapping.items()}
    normalized = copy.deepcopy(report)
    for split in inputs:
        normalized["inputs"][split]["sha256"] = original["inputs"][split]["sha256"]
    for row in normalized["removed_fitting_rows"]:
        for path_key, split_key in (("path", "split"), ("same_as_path", "same_as_split")):
            role = roots[row[split_key]]
            old_path = inverse[(role / row[path_key]).as_posix()]
            row[path_key] = _relative(old_path).relative_to(role).as_posix()
    if normalized != original:
        raise ValueError("Reconstructed known audit differs beyond filenames/input manifest hashes")
    content = (baseline["known_audit"]["content"] if report == original
               else json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return content, report


def _protect(root, protected):
    for name, digest in protected.items():
        path = _safe_path(root, name)
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise ValueError("Protected taxonomy/configuration changed or is missing: " + name)


def plan_refresh(project_root, baseline):
    """Read-only plan; programmatic callers supply a trusted baseline mapping."""
    root = _project_root(project_root)
    _validate_baseline(baseline)
    _protect(root, baseline["protected_files"])
    current = scan_images(root, baseline)
    mapping = build_mapping(baseline["raw_files"], current)
    outputs = render_manifests(baseline, mapping)
    audit_content, known = _known_audit(baseline, outputs, mapping, current)
    outputs[baseline["known_audit"]["path"]] = audit_content
    baseline_outputs = {name: entry["content"] for name, entry in baseline["manifests"].items()}
    baseline_outputs[baseline["known_audit"]["path"]] = baseline["known_audit"]["content"]
    originals, changed = {}, []
    for name in sorted(outputs):
        existing = _read_optional(root, name)
        if existing is not None and existing not in (baseline_outputs[name], outputs[name]):
            raise ValueError("Live manifest/audit has unrelated edits; refusing to overwrite: " + name)
        originals[name] = existing
        if existing != outputs[name]:
            changed.append(name)
    renamed = [dict(old=old, new=new) for old, new in sorted(mapping.items()) if old != new]
    report = dict(schema_version="h02_filename_refresh_report_v1", applied=False,
        image_count=len(current), unique_image_contents=len({r["sha256"] for r in current}),
        manifest_count=len(baseline["manifests"]), renamed_image_count=len(renamed),
        changed_output_count=len(changed), changed_output_files=changed,
        changed_paths=renamed[:10], additional_changed_paths=max(0, len(renamed) - 10),
        known_original_counts=known["original_counts"], known_retained_counts=known["retained_counts"],
        unique_test_known_images=known["unique_test_known_images"], backup_directory=None,
        messages=["Filename-only refresh: image bytes, parent directories, split order, labels and indices are preserved.",
                  "If manifest bytes change, previous completed-run bindings become stale; use a new run directory.",
                  "Images, taxonomy, configuration and existing runs are never rewritten."])
    return dict(project_root=str(root), source_commit=baseline["source_commit"], outputs=outputs,
        originals=originals, mapping=mapping, changed_paths=renamed, changed_output_files=changed,
        protected_files=copy.deepcopy(baseline["protected_files"]), scanned_images=current, report=report)


def _write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def apply_plan(project_root, plan):
    """Stage, back up and replace outputs, restoring original files on failure."""
    root = _project_root(project_root)
    if str(root) != plan["project_root"]:
        raise ValueError("Plan belongs to a different project root")
    _protect(root, plan["protected_files"])
    for name, original in plan["originals"].items():
        if _read_optional(root, name) != original:
            raise ValueError("Output changed since the preview; build a fresh plan: " + name)
    for image in plan["scanned_images"]:
        path = _safe_path(root, image["path"])
        if not path.is_file() or _stat_token(path.stat()) != image["stat"]:
            raise ValueError("Image changed since hashing; build a fresh plan: " + image["path"])
    report = copy.deepcopy(plan["report"])
    report["applied"] = True
    changed = sorted(plan["changed_output_files"])
    if not changed:
        return report
    prepro = _safe_path(root, "prepro")
    backup_parent = _safe_path(root, "prepro/backups")
    created_dirs, replaced = [], []
    with tempfile.TemporaryDirectory(prefix=".h02_filenames_stage_", dir=str(prepro)) as temporary:
        staged = Path(temporary)
        for index, name in enumerate(changed):
            (staged / str(index)).write_bytes(plan["outputs"][name].encode("utf-8"))
        backup_parent.mkdir(parents=True, exist_ok=True)
        prefix = "h02_filenames_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_"
        backup = Path(tempfile.mkdtemp(prefix=prefix, dir=str(backup_parent)))
        report["backup_directory"] = str(backup)
        for name in changed:
            original = plan["originals"][name]
            if original is not None:
                destination = backup / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(original.encode("utf-8"))
        _write_json(backup / "plan.json", plan)
        try:
            for index, name in enumerate(changed):
                target = _safe_path(root, name)
                if _read_optional(root, name) != plan["originals"][name]:
                    raise ValueError("Output changed during apply: " + name)
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created_dirs.append(parent)
                os.replace(str(staged / str(index)), str(target))
                replaced.append(name)
            _write_json(backup / "result.json", dict(status="completed", report=report))
        except BaseException as exc:
            failures = []
            for index, name in enumerate(reversed(replaced)):
                try:
                    target = _safe_path(root, name)
                    original = plan["originals"][name]
                    if original is None:
                        target.unlink()
                    else:
                        restore = staged / ("restore_" + str(index))
                        restore.write_bytes(original.encode("utf-8"))
                        os.replace(str(restore), str(target))
                except BaseException as rollback_error:
                    failures.append(name + ": " + str(rollback_error))
            for parent in reversed(created_dirs):
                try:
                    parent.rmdir()
                except OSError:
                    pass
            _write_json(backup / "result.json", dict(status="rollback_failed" if failures else "rolled_back",
                error=str(exc), rollback_errors=failures))
            if failures:
                raise RuntimeError("Apply failed; restore from backup %s: %s" % (backup, "; ".join(failures))) from exc
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--apply", action="store_true", help="Back up and write the verified plan; default is dry-run")
    args = parser.parse_args(argv)
    try:
        baseline = load_baseline(args.baseline)
        plan = plan_refresh(args.project_root, baseline)
        report = apply_plan(args.project_root, plan) if args.apply else plan["report"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "Filename refresh refused: " + str(exc) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
