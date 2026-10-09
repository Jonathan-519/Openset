"""Audited full spatial grids from the exact trained reference encoder.

TRAIN/DEV images are checked before model construction. TEST image decoding
occurs only after the suite's immutable development decision exists.
"""
import copy
from pathlib import Path
import time

import torch
from PIL import Image

from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_refine.importer import load_reference
from taxosafe_discovery.backend import tensor_hash
from .matching import grid_positions
from . import protocol

SCHEMA_VERSION = "morphology_reference_spatial_v1"


def audited_rows(cache, info, decode=True):
    """Return ALL missing/corrupt images rather than silently shrinking a split."""
    groups, problems, inventory, seen = {}, [], {}, set()
    for split, group in cache["groups"].items():
        rows = {row["image_sha256"]: row for row in base.unique_records(group["records"])}
        aligned = []
        for digest in group["image_sha256"]:
            row = copy.deepcopy(rows[digest])
            if digest in seen:
                raise ValueError("Spatial cache content overlaps splits")
            seen.add(digest)
            path = Path(row["resolved_path"])
            if not path.is_file():
                root_key = "ood_dev_root" if split == "val_extra" else "ood_test_root" if split == "test_extra" else "data_root"
                relocated = protocol.resolve(info["reference"]["config"]["data"][root_key]) / row["path"]
                if relocated.is_file():
                    path = relocated
            try:
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                if protocol.file_hash(path) != digest:
                    raise ValueError("Image bytes differ from frozen content hash")
                if decode:
                    with Image.open(path) as image:
                        image.load()
                        if image.width < 1 or image.height < 1:
                            raise ValueError("Empty image")
                        size = list(image.size)
                else:
                    size = None
                row["resolved_path"] = str(path.resolve())
                inventory[digest] = dict(path=str(path), original_size=size)
            except (OSError, ValueError) as error:
                problems.append(dict(split=split, path=str(path), image_sha256=digest, error=str(error)))
            aligned.append(row)
        groups[split] = aligned
    return groups, dict(valid=not problems, image_count=len(seen),
                       checked_image_count=len(inventory), problems=problems, inventory=inventory,
                       test_images_opened=any(k.startswith("test_") for k in groups))


@torch.no_grad()
def collect_spatial(cache, info, settings, device="cpu"):
    groups, audit = audited_rows(cache, info)
    if not audit["valid"]:
        raise ValueError("Missing/changed/unreadable raw images; restore originals, never substitute TEST: " +
                         str(audit["problems"]))
    started = time.perf_counter()
    source = load_reference(info["reference"]["directory"], torch.device("cpu"))
    if source.binding != info["reference"]["binding"]:
        raise ValueError("Spatial encoder source changed after preflight")
    encoder = source.encoder.to(device).eval().requires_grad_(False)
    cfg = copy.deepcopy(source.config)
    cfg["data"].update(eval_batch_size=settings["batch_size"], n_workers=settings["workers"])
    output, image_count = {}, 0
    for split, rows in groups.items():
        chunks, seen, max_gap = [], [], 0.
        loader = support.make_loader(rows, cfg, source.meta, training=False)
        expected = cache["groups"][split]["features"]
        for images, _, indices in loader:
            global_features, spatial = encoder.backbone.encode_image_with_spatial(images.to(device), normalize=True)
            parent, _ = encoder.parent_branch(global_features, spatial)
            fine, _ = (parent, None) if encoder.shared_encoder else encoder.fine_branch(global_features, spatial)
            for actual, name in ((parent, "source_parent"), (fine, "source_fine")):
                gap = float((actual.detach().cpu().float() - expected[name][indices].float()).abs().max())
                max_gap = max(max_gap, gap)
            if spatial.ndim != 3 or not bool(torch.isfinite(spatial).all()):
                raise ValueError("Reference encoder returned invalid complete spatial tokens")
            chunks.append(spatial.detach().cpu().float())
            seen.extend(indices.tolist())
        if seen != list(range(len(rows))) or max_gap > 1e-4:
            raise ValueError("Spatial input does not reproduce the frozen reference features: " + str(max_gap))
        tokens = torch.cat(chunks)
        positions = grid_positions(tokens.shape[1])
        output[split] = dict(tokens=tokens, positions=positions,
                             image_sha256=list(cache["groups"][split]["image_sha256"]),
                             token_sha256=tensor_hash(tokens), positions_sha256=tensor_hash(positions),
                             reference_feature_max_abs_gap=max_gap)
        image_count += len(rows)
        print("Spatial cache {}: {} images, {} tokens/image, reference gap {:.3g}".format(
            split, len(rows), tokens.shape[1], max_gap), flush=True)
    result = dict(schema_version=SCHEMA_VERSION, groups=output, settings=copy.deepcopy(settings),
                  reference_binding=copy.deepcopy(source.binding), audit=audit,
                  feature_semantics="trained reference backbone complete spatial grid, before TokenBranch pooling",
                  view="exact reference evaluation transform", no_background_or_patch_removal=True,
                  frozen_encoder_updated=False, image_forward_count=image_count,
                  seconds=time.perf_counter()-started)
    encoder.cpu()
    return result


def validate_spatial(spatial, cache, info, settings):
    if (spatial.get("schema_version") != SCHEMA_VERSION or spatial.get("settings") != settings
            or spatial.get("reference_binding") != info["reference"]["binding"]
            or set(spatial.get("groups", {})) != set(cache["groups"])
            or not spatial.get("audit", {}).get("valid")):
        raise ValueError("Spatial cache source/settings/split mismatch")
    for split, original in cache["groups"].items():
        current = spatial["groups"][split]
        tokens, positions = current["tokens"], current["positions"]
        if (tokens.ndim != 3 or len(tokens) != len(original["image_sha256"])
                or tokens.dtype != torch.float32 or not bool(torch.isfinite(tokens).all())
                or current["image_sha256"] != original["image_sha256"]
                or not torch.equal(positions, grid_positions(tokens.shape[1]))
                or tensor_hash(tokens) != current["token_sha256"]
                or tensor_hash(positions) != current["positions_sha256"]):
            raise ValueError("Spatial tensor identity changed: " + split)
    return spatial


def spatial_summary(spatial):
    """Text audit survives review packaging without distributing token tensors."""
    return dict(schema_version=spatial["schema_version"], settings=spatial["settings"],
                reference_binding=spatial["reference_binding"],
                feature_semantics=spatial["feature_semantics"], view=spatial["view"],
                frozen_encoder_updated=False, no_background_or_patch_removal=True,
                groups={split:dict(shape=list(group["tokens"].shape),
                    token_sha256=group["token_sha256"],positions_sha256=group["positions_sha256"],
                    reference_feature_max_abs_gap=group["reference_feature_max_abs_gap"])
                    for split,group in spatial["groups"].items()})
