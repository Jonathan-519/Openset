"""Small, explicit TaxoSieve artifact contract. Historical receipts are never edited."""
import copy
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform

import yaml

from taxosafe_support.io import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)

SCHEMA = "taxosieve_v1"
DEFAULT_CONFIG = "configs/taxosieve.yml"
DEFAULTS = {
    "version": "TaxoSieve_v1",
    "reference_config": "configs/taxosieve_reference.yml",
    "d05": dict(seed=1, folds=3, epochs=100, batch_size=1024, lr=.001,
                shrinkage=.1, hidden=32),
    "d05_calibration": dict(seed=1, shrinkage=20., min_parent_known=5),
    "calibration": dict(seed=1, root_known_target=.92, root_near_target=.85),
}


def runtime_versions():
    """Record the actual installed packages for each new run."""
    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "scipy",
                 "scikit-learn", "ftfy", "regex", "tqdm", "packaging"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def effective_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    # This entry point intentionally reproduces one experiment. New ablations
    # belong in comparison_experiments, with a distinct identity and receipt.
    if object_hash(cfg) != object_hash(DEFAULTS):
        raise ValueError("TaxoSieve settings differ from the locked TaxoSieve_v1 recipe")
    return copy.deepcopy(cfg)


def code_signature():
    files = {p for name in ("taxosieve", "taxosafe_support", "models", "loader")
             for p in (PROJECT_ROOT / name).glob("*.py")}
    files.update(PROJECT_ROOT / name for name in (
        "run_taxosieve.py", "taxosafe_episode.py", "metrics_open.py",
        "models/bpe_simple_vocab_16e6.txt.gz"))
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in sorted(files)})


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular artifact: " + str(path))
    return path


def artifact(directory, descriptor):
    if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
            or not isinstance(descriptor["path"], str)
            or Path(descriptor["path"]).name != descriptor["path"]
            or descriptor["path"] in ("", ".", "..")):
        raise ValueError("Invalid local artifact descriptor")
    path = regular(Path(directory) / descriptor["path"])
    if file_hash(path) != descriptor["sha256"]:
        raise ValueError("Artifact digest mismatch: " + str(path))
    return path


def initialize(directory, cfg, reference_directory, device, mode="train"):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Use an absent run directory, or continue its completed stages: " + str(directory))
    reference_directory = Path(reference_directory).resolve()
    if mode not in ("train", "legacy_d05_import") or device not in ("cuda", "cpu"):
        raise ValueError("Unsupported TaxoSieve source mode or image-inference device")
    from taxosafe_support.protocol import effective_config as reference_config
    reference = reference_config(resolve(cfg["reference_config"]), seed=cfg["d05"]["seed"])
    value = dict(schema_version=SCHEMA, version=cfg["version"], config=cfg,
        code_sha256=code_signature(), reference_directory=str(reference_directory),
        reference_config_sha256=object_hash(reference), device=device, mode=mode,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), runtime_versions=runtime_versions(), test_used_for_fitting=False,
        evaluation_order="reference_then_D05_TRAIN_then_DEV_staged_and_OOF_then_frozen_TEST")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "run.json", value)
    return value


def inspect_run(directory):
    directory = Path(directory).resolve()
    value = read_json(regular(directory / "run.json"))
    if (value.get("schema_version") != SCHEMA or value.get("config") != DEFAULTS
            or value.get("version") != DEFAULTS["version"]
            or value.get("code_sha256") != code_signature()
            or value.get("test_used_for_fitting") is not False
            or value.get("mode") not in ("train", "legacy_d05_import")
            or value.get("device") not in ("cpu", "cuda")):
        raise ValueError("TaxoSieve run identity, configuration or executing code changed")
    from taxosafe_support.protocol import effective_config as reference_config
    reference = reference_config(resolve(value["config"]["reference_config"]),
                                 seed=value["config"]["d05"]["seed"])
    if value.get("reference_config_sha256") != object_hash(reference):
        raise ValueError("Reference recipe changed after TaxoSieve initialization")
    return value


def save_source(directory, reference_binding, legacy=None):
    path = Path(directory) / "source.json"
    value = dict(schema_version=SCHEMA, reference_binding=copy.deepcopy(reference_binding),
                 legacy=copy.deepcopy(legacy))
    if path.exists():
        if read_json(regular(path)) != value:
            raise ValueError("Frozen TaxoSieve source binding changed")
    else:
        write_json(path, value)
    return value


def inspect_source(directory):
    from .source import inspect_reference
    run = inspect_run(directory)
    source = read_json(regular(Path(directory) / "source.json"))
    reference = inspect_reference(run["reference_directory"])
    if (source.get("schema_version") != SCHEMA
            or source.get("reference_binding") != reference["binding"]
            or object_hash(reference["config"]) != run["reference_config_sha256"]):
        raise ValueError("Reference artifacts or locked configuration changed")
    legacy = source.get("legacy")
    if legacy is not None:
        # Recheck the old bytes even when using the new compact run. Paths are
        # concrete files verified by the original importer during export.
        for name, digest in legacy["files"].items():
            if file_hash(regular(name)) != digest:
                raise ValueError("Imported D05 source changed: " + name)
    elif run["mode"] == "legacy_d05_import":
        raise ValueError("The imported D05 source evidence is missing")
    return reference, source


def stage_path(directory, stage):
    if stage not in ("cache/train", "cache/development", "cache/test", "training", "calibration", "test"):
        raise ValueError("Unknown TaxoSieve stage")
    return Path(directory) / stage


def claim_stage(directory, stage):
    path = stage_path(directory, stage)
    if path.exists():
        raise ValueError("Stage is partial or already exists; it is never overwritten: " + str(path))
    path.mkdir(parents=True, exist_ok=False)
    return path


def finish_stage(directory, stage, details, names):
    path = stage_path(directory, stage)
    if (path / "completed.json").exists():
        raise ValueError("Cannot replace a completed TaxoSieve stage")
    forbidden = {"schema_version", "version", "stage", "run_sha256", "source_sha256",
                 "artifacts", "test_used_for_fitting", "unknown_images_used_for_gradients"}
    if forbidden.intersection(details):
        raise ValueError("Stage details may not override its provenance contract")
    receipt = dict(schema_version=SCHEMA, version=DEFAULTS["version"], stage=stage,
        run_sha256=file_hash(regular(Path(directory) / "run.json")),
        source_sha256=file_hash(regular(Path(directory) / "source.json")),
        test_used_for_fitting=False, unknown_images_used_for_gradients=False,
        artifacts={key: dict(path=name, sha256=file_hash(regular(path / name)))
                   for key, name in names.items()}, **details)
    write_json(path / "completed.json", receipt)
    return verify_stage(directory, stage)


def verify_stage(directory, stage):
    path = stage_path(directory, stage)
    receipt = read_json(regular(path / "completed.json"))
    expected = dict(schema_version=SCHEMA, version=DEFAULTS["version"], stage=stage,
        run_sha256=file_hash(regular(Path(directory) / "run.json")),
        source_sha256=file_hash(regular(Path(directory) / "source.json")),
        test_used_for_fitting=False, unknown_images_used_for_gradients=False)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage source/run/permission identity changed: " + stage)
    if not isinstance(receipt.get("artifacts"), dict) or not receipt["artifacts"]:
        raise ValueError("A completed stage must bind its actual artifacts")
    for descriptor in receipt["artifacts"].values():
        artifact(path, descriptor)
    return receipt


def completed(directory, stage):
    path = stage_path(directory, stage)
    if not path.exists():
        return False
    verify_stage(directory, stage)
    return True
