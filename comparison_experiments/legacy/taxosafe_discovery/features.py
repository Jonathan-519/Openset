"""Audited, content-deduplicated reference and prompt-free CLIP image caches.

The frozen pretrained core is reconstructed from the reference's unchanged
core tensors. No download, learned MaPLe context or historical TokenBranch is
used in the vanilla path. Source output uses the historical inference exactly.
"""
import copy
import gc
import hashlib
import math
import time

import torch
from torch.nn import functional as F

from taxosafe_refine.pipeline import _frozen
from taxosafe_support import pipeline as support
from taxosafe_support.calibration import raw_records, unique_records
from taxosafe_support.membership_calibration import RAW_FIELDS
from taxosafe_support.protocol import object_hash
from .models import clip_image_features


SINGLE_TEMPLATES = ("a photo of a {}.",)
ENSEMBLE_TEMPLATES = (
    "a microscopy image of a {}.",
    "a microscope image of a {} specimen.",
    "an image of a {} in a plankton sample.",
    "a microscopic view of a {}.",
)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def core_tensor_hash(core):
    return object_hash({key: tensor_hash(value) for key, value in sorted(core.state_dict().items())})


def build_clip_core(source, device="cpu"):
    """Strictly rebuild a vanilla CLIP ViT from frozen, unprompted source tensors."""
    from models.model import CLIP
    _frozen(source)
    backbone = source.encoder.backbone
    model = getattr(backbone, "model", None)
    visual = getattr(model, "image_encoder", None)
    text = getattr(model, "text_encoder", None)
    token_embedding = getattr(backbone, "token_embedding", None)
    if (visual is None or text is None or token_embedding is None
            or not hasattr(visual, "proj") or not hasattr(visual, "conv1")):
        raise ValueError("Prompt-free reconstruction requires a supported frozen MaPLe ViT core")
    state = {"visual." + key: value.detach().cpu().clone() for key, value in visual.state_dict().items()}
    state.update({key: value.detach().cpu().clone() for key, value in text.state_dict().items()})
    state["token_embedding.weight"] = token_embedding.weight.detach().cpu().clone()
    state["logit_scale"] = model.logit_scale.detach().cpu().clone()
    if any("prompt" in key.lower() or "vpt" in key.lower() for key in state):
        raise ValueError("Prompt tensors must not enter the vanilla CLIP core")
    patch_size = state["visual.conv1.weight"].shape[-1]
    patch_count = state["visual.positional_embedding"].shape[0] - 1
    grid = math.isqrt(patch_count)
    if grid * grid != patch_count:
        raise ValueError("CLIP positional grid must be square")
    width = state["visual.conv1.weight"].shape[0]
    text_width = state["ln_final.weight"].numel()
    if width % 64 or text_width % 64:
        raise ValueError("Unsupported vanilla CLIP attention width")
    vision_layers = len([key for key in state if key.startswith("visual.transformer.resblocks.")
                         and key.endswith("attn.in_proj_weight")])
    text_layers = len([key for key in state if key.startswith("transformer.resblocks.")
                       and key.endswith("attn.in_proj_weight")])
    # Constructor initialization must not advance the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        core = CLIP(int(state["text_projection"].shape[1]), grid * patch_size,
                    vision_layers, width, patch_size, int(state["positional_embedding"].shape[0]),
                    int(state["token_embedding.weight"].shape[0]), text_width,
                    text_width // 64, text_layers).float()
    core.load_state_dict(state, strict=True)
    for key, value in core.state_dict().items():
        if not torch.equal(value.cpu(), state[key].to(value.dtype)):
            raise ValueError("Frozen core reconstruction changed tensor: " + key)
    return core.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_templates(core, names, templates):
    """Normalize each template embedding, average templates, then normalize."""
    from models.clip import tokenize
    if (not names or not templates or any(not isinstance(name, str) or not name.strip() for name in names)
            or any(not isinstance(t, str) or t.count("{}") != 1 for t in templates)):
        raise ValueError("Templates require nonempty names and exactly one placeholder")
    device = next(core.parameters()).device
    names = [name.replace("_", " ") for name in names]
    encoded = []
    for template in templates:
        tokens = tokenize([template.format(name) for name in names], context_length=core.context_length).to(device)
        encoded.append(F.normalize(core.encode_text(tokens).float(), dim=-1))
    result = F.normalize(torch.stack(encoded).mean(0), dim=-1).cpu()
    if not bool(torch.isfinite(result).all()) or bool((result.norm(dim=-1) < .99).any()):
        raise ValueError("Text ensemble produced invalid normalized features")
    return result


def _module_device(module):
    return next(module.parameters()).device


@torch.no_grad()
def collect_cache(source, groups, device):
    """Collect only supplied splits; feature rows are unique, records retain aliases.

    ``record_feature_indices`` explicitly maps each raw record to a unique
    feature row. Original baseline evidence deliberately preserves original
    support inference; new episode fitting must perform its own self exclusion.
    """
    _frozen(source)
    if not groups:
        raise ValueError("Feature collection needs nonempty supplied splits")
    device = torch.device(device)
    original = (_module_device(source.encoder), _module_device(source.evidence), source.bank.parent.device)
    canonical, cache_groups, timings, seen_hashes = {}, {}, {}, set()
    for split, raw in groups.items():
        rows = list(raw)
        unique = unique_records(rows)
        if not unique or any(row.get("split") != split for row in unique):
            raise ValueError("Empty or mislabeled discovery split: " + split)
        hashes = {row["image_sha256"] for row in unique}
        if hashes & seen_hashes:
            raise ValueError("Discovery image content overlaps supplied splits")
        seen_hashes.update(hashes)
        canonical[split] = unique
        indices = {row["image_sha256"]: i for i, row in enumerate(unique)}
        cache_groups[split] = {"records": rows, "features": {},
                               "image_sha256": list(indices),
                               "record_feature_indices": [indices[row["image_sha256"]] for row in rows]}
    clip_core = None
    try:
        source.encoder.to(device)
        source.evidence.to(device)
        source.bank.to(device)
        source_text = source.encoder.text_features()
        for split, unique in canonical.items():
            started = time.perf_counter()
            seen, scored, fine, parent = [], [], [], []
            for images, _, indices in support.make_loader(unique, source.config, source.meta):
                encoded = source.encoder.encode(images.to(device), text_features=source_text)
                output = source.evidence(encoded, source.bank)
                selected = [unique[i] for i in indices.tolist()]
                seen.extend(indices.tolist())
                batch = raw_records(selected, {"log_probs": output["log_probs"].cpu().numpy()},
                                    encoded["leaf_logits"].cpu().numpy(), source.meta)
                diagnostic = {key: output[key].cpu().tolist() for key in RAW_FIELDS}
                for i, row in enumerate(batch):
                    row["support_evidence"] = {key: values[i] for key, values in diagnostic.items()}
                scored.extend(batch)
                fine.append(encoded["fine"].float().cpu())
                parent.append(encoded["parent"].float().cpu())
            if seen != list(range(len(unique))):
                raise ValueError("Reference collection must preserve unique manifest order")
            by_hash = {row["image_sha256"]: row for row in scored}
            group = cache_groups[split]
            group["records"] = [dict(by_hash[row["image_sha256"]], **row) for row in group["records"]]
            group["features"].update(source_fine=torch.cat(fine), source_parent=torch.cat(parent))
            timings[split] = {"source_seconds": time.perf_counter() - started,
                              "unique_images": len(unique), "manifest_rows": len(group["records"])}
        # Release transient source outputs before constructing/moving vanilla CLIP.
        del source_text, encoded, output, images
        source.encoder.cpu()
        source.evidence.cpu()
        source.bank.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        clip_core = build_clip_core(source, "cpu")
        core_digest = core_tensor_hash(clip_core)
        clip_core.to(device)
        names = source.meta["leaf_names"] + source.meta["parent_names"]
        count = len(source.meta["leaf_names"])
        single = encode_templates(clip_core, names, SINGLE_TEMPLATES)
        ensemble = encode_templates(clip_core, names, ENSEMBLE_TEMPLATES)
        text = {"single_leaf": single[:count], "single_parent": single[count:],
                "ensemble_leaf": ensemble[:count], "ensemble_parent": ensemble[count:]}
        for split, unique in canonical.items():
            started = time.perf_counter()
            seen, values = [], []
            for images, _, indices in support.make_loader(unique, source.config, source.meta):
                values.append(clip_image_features(clip_core, images.to(device))["global"].cpu())
                seen.extend(indices.tolist())
            if seen != list(range(len(unique))):
                raise ValueError("CLIP collection must preserve unique manifest order")
            cache_groups[split]["features"]["clip"] = torch.cat(values)
            timings[split].update(clip_seconds=time.perf_counter() - started,
                                  visual_passes_per_unique_image=2)
        for group in cache_groups.values():
            for name, value in group["features"].items():
                if (value.ndim != 2 or len(value) != len(group["image_sha256"])
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("Invalid discovery feature cache: " + name)
        provenance = {"schema_version": "discovery_features_v1", "source_binding": copy.deepcopy(source.binding),
                      "clip_initialization": "source_frozen_pretrained_core_without_prompts_or_adapters",
                      "clip_core_sha256": core_digest, "new_weights_downloaded": False,
                      "single_templates": list(SINGLE_TEMPLATES), "ensemble_templates": list(ENSEMBLE_TEMPLATES),
                      "template_names": list(names), "template_pooling": "mean_of_unit_vectors_then_normalize",
                      "text_sha256": {key: tensor_hash(value) for key, value in text.items()},
                      "preprocessing": copy.deepcopy(source.config["data"]),
                      "baseline_arithmetic": "original_reference_encode_and_evidence_without_new_self_exclusion",
                      "feature_gradients": False, "spatial_tokens_cached": False}
        provenance["templates_sha256"] = object_hash({key: provenance[key] for key in
             ("single_templates", "ensemble_templates", "template_names", "template_pooling", "text_sha256")})
        return {"meta": copy.deepcopy(source.meta), "groups": cache_groups, "text": text,
                "provenance": provenance, "timings": timings}
    finally:
        if clip_core is not None:
            clip_core.cpu()
            del clip_core
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        source.encoder.to(original[0]).eval().requires_grad_(False)
        source.evidence.to(original[1]).eval().requires_grad_(False)
        source.bank.to(original[2])
        _frozen(source)
