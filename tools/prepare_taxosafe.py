"""Prepare the known view and verify data/runtime prerequisites without training.

The vocabulary download is explicit, pinned to official CLIP, and verified
before installation. No model is downloaded and no checkpoint is fabricated.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BPE_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
BPE_URL = ("https://raw.githubusercontent.com/openai/CLIP/" + BPE_COMMIT
           + "/clip/bpe_simple_vocab_16e6.txt.gz")
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
BPE_PATH = ROOT / "models/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_clean_v1.yml"
DEPENDENCIES = {"numpy": "numpy", "yaml": "PyYAML", "sklearn": "scikit-learn",
                "PIL": "Pillow", "torch": "torch", "torchvision": "torchvision",
                "ftfy": "ftfy", "regex": "regex", "tqdm": "tqdm",
                "termcolor": "termcolor", "tensorboardX": "tensorboardX", "packaging": "packaging"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_bpe(path=BPE_PATH, download=False):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not path.is_file() or file_hash(path) != BPE_SHA256:
            raise ValueError("Unexpected vocabulary at {}; inspect/move it manually. "
                             "Existing files are never replaced.".format(path))
        return {"status": "verified", "sha256": BPE_SHA256, "source": BPE_URL}
    if not download:
        raise FileNotFoundError("Missing CLIP vocabulary; run python tools/prepare_taxosafe.py --download-bpe")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(BPE_URL, timeout=30) as source:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as out:
                temporary = Path(out.name)
                # The pinned archive is 1,356,917 bytes; reject unexpected responses.
                content = source.read(2_000_001)
                if len(content) != 1_356_917 or hashlib.sha256(content).hexdigest() != BPE_SHA256:
                    raise ValueError("CLIP vocabulary download failed size/SHA-256 verification")
                out.write(content)
        # Exclusive creation also protects an existing file in a concurrent run.
        with path.open("xb") as target, temporary.open("rb") as source:
            target.write(source.read())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "downloaded_and_verified", "sha256": BPE_SHA256, "source": BPE_URL}


def dependency_report(require_cuda=False):
    versions, errors = {}, []
    for module, package in DEPENDENCIES.items():
        try:
            importlib.import_module(module)
            versions[package] = importlib.metadata.version(package)
        except Exception as error:
            errors.append("{}: {}".format(package, error))
    cuda = {"available": False}
    if "torch" in versions:
        import torch
        cuda = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    if require_cuda and not cuda["available"]:
        errors.append("MaPLe training/extraction requires CUDA; this environment has no usable CUDA device")
    return {"versions": versions, "cuda": cuda, "errors": errors}


def check_data(config):
    """Inspect labels/paths and exact bytes; never fit on validation/test data."""
    import yaml
    from validate_taxosafe_setup import validate
    from audit_taxosafe_splits import audit
    with Path(config).open(encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)
    errors, notes = validate(cfg, skip_image_check=False)
    if errors:
        raise ValueError("\n".join(errors[:30]))
    report = audit(ROOT, config)
    totals = report["totals"]
    missing = [s for s, item in report["split_summaries"].items() if item["status"] != "ok"]
    if missing or any(totals[k] for k in ("cross_split_duplicate_groups",
                                       "within_split_duplicate_groups", "status_or_label_conflict_groups")):
        raise ValueError("Incomplete or duplicated data: missing={}, totals={}".format(missing, totals))
    return {"totals": totals, "split_counts": {
        s: item["record_count"] for s, item in report["split_summaries"].items()}, "notes": notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Training YAML, not the visual extension")
    parser.add_argument("--download-bpe", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Do not create the known view or download files")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", help="Optional JSON output path")
    args = parser.parse_args()
    if args.check_only and args.download_bpe:
        parser.error("--check-only cannot be combined with --download-bpe")
    os.chdir(ROOT)
    report = {"python": sys.version.split()[0], "errors": []}
    if not args.check_only:
        from prepro.build_known_view import build_known_view
        try:
            report["created_species_links"] = build_known_view(
                ROOT / "prepro/raw/Zooplankton_Taxonomic_Tree",
                ROOT / "prepro/splits/Zooplankton_Taxonomic_Tree/fold1.json",
                ROOT / "prepro/views/Zooplankton/fold1_known")
        except Exception as error:
            report["errors"].append(str(error))
    try:
        report["vocabulary"] = ensure_bpe(download=args.download_bpe)
    except Exception as error:
        report["errors"].append(str(error))
    report["runtime"] = dependency_report(args.require_cuda)
    report["errors"].extend(report["runtime"]["errors"])
    try:
        report["data"] = check_data(args.config)
    except Exception as error:
        report["errors"].append("Data: {}".format(error))
    if not report["errors"]:
        try:
            import yaml
            import train_taxosafe
            from loader.hierdata import _load_hierarchy
            from models.simple_tokenizer import SimpleTokenizer
            with open(args.config, encoding="utf-8-sig") as stream:
                cfg = yaml.safe_load(stream)
            hierarchy = _load_hierarchy(cfg["data"])
            import torch
            meta = train_taxosafe.build_hier_meta(
                hierarchy["param_names"], hierarchy["leaf_nodes"], hierarchy["intnl_nodes"],
                hierarchy["sublabels"], torch.device("cpu"), cfg["data"]["num_known_leaves"])
            report["hierarchy"] = {"parents": len(meta["parent_names"]), "leaves": len(meta["leaf_names"])}
            report["tokenizer_vocab_size"] = len(SimpleTokenizer().encoder)
        except Exception as error:
            report["errors"].append("Entry point/hierarchy import: {}".format(error))
    report["ready_for_cuda_training"] = not report["errors"] and report["runtime"]["cuda"]["available"]
    report["trained_checkpoint_verified"] = False
    report["note"] = "Preparation only. CLIP backbone is downloaded by MaPLe on first training use; no trained TaxoSafe checkpoint is bundled."
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
