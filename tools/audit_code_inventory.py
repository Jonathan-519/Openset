"""Read-only Python inventory, duplicate-body and configured-path audit.

Unlike the historical data audit, this command never imports project modules,
reads image pixels or requires old manifests to exist. Candidate duplicates and
unused imports are review leads, not a safe-to-delete list. Frozen historical
source files must keep their bytes when existing experiment receipts bind them.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def repository_paths(root):
    """List Git paths or a pruned source tree when run from an exported ZIP."""
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        paths = []
        skipped = {"raw", "runs", ".git", "venv", ".venv", "__pycache__", "weights"}
        for folder, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(name for name in directories
                                    if name not in skipped and not name.endswith("-venv")
                                    and not (Path(folder) / name).is_symlink())
            paths.extend((Path(folder) / name).relative_to(root).as_posix() for name in files)
        return sorted(paths), "filesystem"
    return sorted(set(raw.decode("utf-8").split("\0")) - {""}), "git"


def _checked_out(root, name):
    path = root / name
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return path.is_file() and not path.is_symlink()


def _main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            and any(isinstance(v, ast.Constant) and v.value == "__main__"
                    for v in node.test.comparators))


def _source_summary(name, source):
    tree = ast.parse(source, filename=name)
    loads = {node.id for node in ast.walk(tree)
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    imports, candidates = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append("." * node.level + (node.module or ""))
    # Restrict candidates to top-level imports; exports, decorators, registration
    # and import-time side effects still require human inspection.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name != "*" and bound != "annotations" and bound not in loads:
                    candidates.append({"name": bound, "line": node.lineno})
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.end_lineno - node.lineno < 5 or len(node.body) < 2:
                continue
            normalized = copy.deepcopy(node)
            normalized.name = "_function_name"
            functions.append((ast.dump(normalized, include_attributes=False),
                              {"file": name, "function": node.name,
                               "line": node.lineno, "end_line": node.end_lineno}))
    return {"lines": len(source.splitlines()), "imports": sorted(set(imports)),
            "definitions": [n.name for n in tree.body
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
            "cli_entrypoint": any(_main_guard(node) for node in tree.body),
            "unused_import_candidates": candidates}, functions


def _config_references(root, paths):
    reports = {}
    for name in paths:
        if not name.startswith("configs/") or Path(name).suffix not in (".yml", ".yaml"):
            continue
        if not _checked_out(root, name):
            reports[name] = {"status": "not_checked_out", "references": []}
            continue
        try:
            cfg = yaml.safe_load((root / name).read_text(encoding="utf-8-sig"))
            if not isinstance(cfg, dict):
                raise ValueError("Configuration root is not a mapping")
            data = cfg.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Configuration data section is not a mapping")
            refs = []
            for key, value in sorted(data.items()):
                # Inspect explicitly declared paths without guessing an old
                # full_data_root/ood_root schema for v11/new configurations.
                if isinstance(value, str) and ("/" in value or "\\" in value):
                    path = Path(value)
                    candidate = path if path.is_absolute() else root / path
                    refs.append({"key": key, "path": value,
                                 "exists_locally": candidate.exists()})
            reports[name] = {"status": "ok", "references": refs}
        except (OSError, ValueError, yaml.YAMLError) as exc:
            reports[name] = {"status": "error", "error": str(exc), "references": []}
    return reports


def audit(root=ROOT):
    root = Path(root).resolve()
    paths, inventory_mode = repository_paths(root)
    sources, errors, missing, duplicates = {}, [], [], defaultdict(list)
    for name in paths:
        if Path(name).suffix != ".py":
            continue
        if not _checked_out(root, name):
            missing.append(name)
            continue
        try:
            summary, functions = _source_summary(name, (root / name).read_text(encoding="utf-8-sig"))
            sources[name] = summary
            for body, location in functions:
                duplicates[body].append(location)
        except (SyntaxError, UnicodeError, OSError) as exc:
            errors.append({"file": name, "error": str(exc)})
    root_test_clis = [name for name, summary in sources.items()
                     if "/" not in name and Path(name).match("test*.py") and summary["cli_entrypoint"]]
    groups = Counter(name.split("/", 1)[0] if "/" in name else "<root>" for name in sources)
    base_commit = None
    if inventory_mode == "git":
        try:
            base_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass  # A newly initialized repository may not have a commit yet.
    return {
        "base_commit": base_commit, "inventory_mode": inventory_mode,
        "scope": "Working-tree Python AST and config path declarations; tracked sparse paths retained. No imports, image pixels, checkpoint or manifest contents read.",
        "inventory_paths": len(paths), "python_sources_parsed": len(sources),
        "source_lines": sum(item["lines"] for item in sources.values()),
        "groups": dict(sorted(groups.items())), "syntax_errors": errors,
        "python_not_checked_out": missing, "sources": sources,
        "duplicate_function_groups": [locations for locations in duplicates.values() if len(locations) > 1],
        "config_references": _config_references(root, paths),
        "test_discovery": {"root_inference_cli_modules": root_test_clis,
                           "recommended_command": "PYTHONPATH=.:tests python -m unittest discover -s tests -t . -v",
                           "note": "Default repository-root discovery also imports matching inference CLIs and their model/runtime dependencies; these are not unit tests."},
        "review_warning": "Unused-import and duplicate-body candidates do not prove dead code. Keep public exports, old algorithm entrypoints and all receipt-bound files unless versioning the protocol explicitly.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; never overwritten")
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "base_commit", "inventory_mode", "python_sources_parsed", "source_lines", "groups",
        "syntax_errors", "python_not_checked_out", "test_discovery")}, ensure_ascii=False, indent=2))
    if args.output:
        print("Audit:", args.output)


if __name__ == "__main__":
    main()
