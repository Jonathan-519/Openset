"""Read-only tracked-file/import/manifest audit; works with a sparse clone.

Git blob IDs identify identical tracked bytes, not perceptual duplicates.
Run: python tools/audit_repository.py --output docs/repository_audit.json
No images or manifests are deleted or rewritten.
"""
import argparse
import ast
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def inventory():
    raw = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"], cwd=str(ROOT))
    files = {}
    for item in raw.decode("utf-8").split("\0"):
        if not item:
            continue
        header, path = item.split("\t", 1)
        mode, kind, sha = header.split()
        if kind == "blob":
            files[path] = {"blob": sha, "kind": "image" if Path(path).suffix.lower() in IMAGE_SUFFIXES else "metadata_or_code"}
    return files


def audit(files):
    sources, errors = {}, []
    for name in sorted(files):
        path = ROOT / name
        if path.suffix != ".py":
            continue
        if not path.is_file():
            errors.append({"file": name, "error": "not checked out"})
            continue
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=name)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append("." * node.level + (node.module or ""))
            sources[name] = {"lines": len(source.splitlines()), "imports": sorted(set(imports)),
                             "definitions": [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]}
        except SyntaxError as exc:
            errors.append({"file": name, "error": str(exc)})
    manifests, all_referenced = {}, set()
    for config in sorted((ROOT / "configs").rglob("*.yml")):
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        if "data" not in cfg:
            continue
        data = cfg["data"]
        records, missing, summaries, view_aliases = [], [], {}, []
        for split in ("train", "val_known", "test_known", "oe_train", "val_intra", "test_intra", "val_extra", "test_extra"):
            if not data.get(split):
                continue
            kind = "known" if split in ("train", "val_known", "test_known") else ("intra" if "intra" in split else "extra")
            root = data[{"known": "data_root", "intra": "full_data_root", "extra": "ood_root"}[kind]]
            counts = Counter()
            for line_no, line in enumerate((ROOT / data[split]).read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                relative, label, _ = line.rsplit(",", 2)
                path = (ROOT / root / relative).resolve().relative_to(ROOT).as_posix()
                view_prefix = "prepro/views/Zooplankton/fold1_known/"
                if path.startswith(view_prefix) and path not in files:
                    raw_path = "prepro/raw/Zooplankton_Taxonomic_Tree/" + path[len(view_prefix):]
                    if raw_path in files:
                        view_aliases.append({"view": path, "raw": raw_path})
                        path = raw_path
                all_referenced.add(path)
                record = {"split": split, "path": path, "label": int(label), "kind": kind, "source": Path(path).parent.name}
                counts[record["source"]] += 1
                if path not in files:
                    missing.append({"split": split, "line": line_no, "path": path})
                else:
                    record["blob"] = files[path]["blob"]
                    records.append(record)
            summaries[split] = {"count": sum(counts.values()), "sources": dict(sorted(counts.items()))}
        by_blob = defaultdict(list)
        for record in records:
            by_blob[record["blob"]].append(record)
        duplicates = [r for r in by_blob.values() if len(r) > 1]
        source_overlap = {}
        for kind in ("intra", "extra"):
            a = set(summaries.get("val_" + kind, {}).get("sources", {}))
            b = set(summaries.get("test_" + kind, {}).get("sources", {}))
            source_overlap[kind] = sorted(a & b)
        manifests[config.relative_to(ROOT).as_posix()] = {
            "splits": summaries, "missing_paths": missing, "exact_duplicate_groups": duplicates,
            "known_view_paths_resolved_to_raw": len(view_aliases),
            "cross_split_duplicate_groups": sum(len({x["split"] for x in r}) > 1 for r in duplicates),
            "validation_test_source_overlap": source_overlap}
    images = {k: v for k, v in files.items() if v["kind"] == "image"}
    duplicate_blobs = defaultdict(list)
    for path, value in files.items():
        duplicate_blobs[value["blob"]].append(path)
    unused_images = sorted(set(images) - all_referenced)
    return {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
            "scope": "All tracked paths and blob IDs; all Python parsed; all configured manifest rows inspected. Image pixels not reviewed.",
            "tracked_count": len(files), "image_count": len(images), "source_count": len(sources),
            "syntax_errors": errors, "sources": sources, "manifests": manifests,
            "duplicate_tracked_files": [p for p in duplicate_blobs.values() if len(p) > 1],
            "images_unreferenced_by_training_configs": unused_images,
            "unreferenced_warning": "Not safe-to-delete automatically: may be raw archive or future-fold data.",
            "all_tracked_files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(inventory())
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new output; audit does not overwrite existing files")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("base_commit", "tracked_count", "image_count", "source_count", "syntax_errors")}, indent=2))
    for name, manifest in result["manifests"].items():
        print(name, "missing:", len(manifest["missing_paths"]), "cross-split duplicate groups:", manifest["cross_split_duplicate_groups"])
        print(json.dumps(manifest["splits"], ensure_ascii=False))
    print("Unreferenced images:", len(result["images_unreferenced_by_training_configs"]))


if __name__ == "__main__":
    main()
