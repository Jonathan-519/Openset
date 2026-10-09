"""Shared legacy evaluation I/O, without importing models or GPU libraries.

Keep UTF-8 encoding, key ordering, trailing newlines and path semantics stable:
existing callers use these helpers when generating hashed experiment artifacts.
The source modules retain compatible wrappers under their original names.
"""
import hashlib
import json
import os

import yaml


def load_yaml(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML root must be a mapping")
    return cfg, path


def resolve_run_dir(cfg, trial, project_root, explicit_run_dir=None):
    if explicit_run_dir:
        return os.path.abspath(explicit_run_dir)
    return os.path.join(
        project_root,
        "runs",
        cfg["data"]["name"],
        cfg["model"]["arch"],
        cfg["exp"],
        "trial_{}".format(trial),
    )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path, records, drop_vector_fields=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            output = dict(record)
            if drop_vector_fields:
                output.pop("parent_cosine", None)
                output.pop("leaf_cosine", None)
                output.pop("image_feature", None)
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
