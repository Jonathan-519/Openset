"""Configuration and provenance for frozen hierarchical relative distances."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, claim_stage, file_hash, object_hash, read_json,
    require_signature, resolve, run_lock, verify_artifact, write_json, write_records,
)

DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_geometry.yml"
SCHEMA_VERSION = "frozen_reference_geometry_v1"


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " must be an integer in [{},{}]".format(low, high))


def _number(value, name, low, high, zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(name + " must be finite and numeric")
    if not math.isfinite(value) or value > high or (value < low if zero else value <= low):
        raise ValueError(name + " is outside its permitted range")


def effective_config(path, seed=None):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or set(cfg) != {"name", "seed", "geometry", "calibration"}:
        raise ValueError("Geometry configuration requires name, seed, geometry and calibration")
    if seed is not None:
        cfg["seed"] = seed
    _integer(cfg["seed"], "seed", 0, 2 ** 31 - 1)
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    geometry = cfg["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {"shrinkage", "ridge"}:
        raise ValueError("Geometry requires shrinkage and ridge")
    _number(geometry["shrinkage"], "geometry.shrinkage", 0., 1., zero=True)
    _number(geometry["ridge"], "geometry.ridge", 0., 1.)
    settings = cfg["calibration"]
    if isinstance(settings, dict) and settings.get("decoder") == "local_guarded":
        from .local import settings as local_settings
        # Baseline calibration settings come from the audited source run only.
        if "baseline_calibration" in settings:
            raise ValueError("baseline_calibration must come from the frozen source")
        local_settings(settings)
        return cfg
    required = {"weights", "grid_points", "source_loo", "source_loo_safeguard"}
    if (not isinstance(settings, dict) or not required <= set(settings)
            or set(settings) - required - {"parent_weights", "leaf_weights"}):
        raise ValueError("Invalid geometry calibration configuration fields")
    _integer(settings["grid_points"], "calibration.grid_points", 2, 101)
    for key in ("source_loo", "source_loo_safeguard"):
        if not isinstance(settings[key], bool):
            raise ValueError("calibration." + key + " must be a YAML boolean")
    if settings["source_loo_safeguard"] and not settings["source_loo"]:
        raise ValueError("source_loo_safeguard requires source_loo")
    for key in ("weights", "parent_weights", "leaf_weights"):
        if key not in settings:
            continue
        values = settings[key]
        if not isinstance(values, list) or not 1 <= len(values) <= 7:
            raise ValueError("calibration." + key + " requires one to seven weights")
        for value in values:
            _number(value, "calibration." + key, 0., 1., zero=True)
        if len(set(values)) != len(values):
            raise ValueError("calibration weights must be distinct")
    return cfg


def signature(cfg, reference_binding):
    paths = sorted((PROJECT_ROOT / "taxosafe_geometry").glob("*.py"))
    # Imported historical code remains unchanged, but is covered by this new
    # method's signature as well as the source's original support-code audit.
    paths += sorted((PROJECT_ROOT / "taxosafe_refine").glob("*.py"))
    paths += [PROJECT_ROOT / "refine_taxosafe_geometry.py", PROJECT_ROOT / "metrics_open.py"]
    return {"method": SCHEMA_VERSION, "config": object_hash(cfg),
            "reference": object_hash(reference_binding),
            "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in paths})}
