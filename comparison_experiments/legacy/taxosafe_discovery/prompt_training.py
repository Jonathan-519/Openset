"""TRAIN-only shallow CoOp/FA controls over frozen vanilla CLIP image caches.

This is a hierarchical, scaled-logit FA adaptation, not a paper reproduction.
Only two shared input contexts have gradients. No development set is accepted.
"""
import copy
import gc
import math

import torch
from torch import nn
from torch.nn import functional as F

from taxosafe_support.calibration import unique_records
from taxosafe_support.protocol import object_hash
from .features import build_clip_core, core_tensor_hash, encode_templates, tensor_hash, SINGLE_TEMPLATES


SCHEMA_VERSION = "discovery_prompt_v1"
DEFAULTS = dict(epochs=20, batch_size=256, lr=.002, n_ctx=4,
                reference_weight=3., temperature=1.)


def _options(options):
    result = dict(DEFAULTS)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(result):
            raise ValueError("Unknown prompt training options")
        result.update(options)
    for key in ("epochs", "batch_size", "n_ctx"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Prompt " + key + " must be a positive integer")
    for key in ("lr", "reference_weight", "temperature"):
        value = result[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Prompt " + key + " must be positive and finite")
    if result["n_ctx"] > 16 or result["lr"] > .01:
        raise ValueError("Prompt context length or learning rate exceeds protocol limits")
    return result


def prompt_objective(logits, labels, reference_logits, loss, reference_weight=3., temperature=1.):
    """CE and FA differ only in the frozen-reference denominator term."""
    if loss not in ("coop", "fa"):
        raise ValueError("Prompt loss must be coop or fa")
    scaled = logits / temperature
    if loss == "coop":
        return F.cross_entropy(scaled, labels)
    reference = reference_logits.detach() / temperature
    denominator = torch.logaddexp(scaled.logsumexp(1),
                                 reference.logsumexp(1) + math.log(reference_weight))
    return (denominator - scaled.gather(1, labels[:, None]).squeeze(1)).mean()


class ShallowPrompts(nn.Module):
    """Frozen text core plus one shared context per hierarchy level.

    No visual module is registered, so moving this module never moves the
    heavyweight visual encoder. Prefix/suffix embeddings remain frozen buffers.
    """
    def __init__(self, core, meta, n_ctx=4):
        super().__init__()
        from models.clip import tokenize
        self.transformer = core.transformer
        self.token_embedding = core.token_embedding
        self.positional_embedding = core.positional_embedding
        self.ln_final = core.ln_final
        self.text_projection = core.text_projection
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.context_text = " ".join((["a", "photo", "of", "a"] + ["X"] * n_ctx)[:n_ctx])
        tokens = tokenize([self.context_text], context_length=core.context_length)
        if int(tokens[0].argmax()) != n_ctx + 1:
            raise ValueError("Prompt initialization must tokenize to n_ctx context tokens")
        with torch.no_grad():
            initial = core.token_embedding(tokens)[0, 1:1 + n_ctx].detach().float()
        self.context = nn.ParameterDict({level: nn.Parameter(initial.clone()) for level in ("leaf", "parent")})
        for level in ("leaf", "parent"):
            names = [name.replace("_", " ") for name in meta[level + "_names"]]
            tokenized = tokenize([self.context_text + " " + name + "." for name in names],
                                 context_length=core.context_length)
            if bool((tokenized.argmax(1) <= n_ctx + 1).any()):
                raise ValueError("Class name is missing from prompt tokens")
            with torch.no_grad():
                embedded = core.token_embedding(tokenized).detach().float()
            self.register_buffer(level + "_prefix", embedded[:, :1].clone())
            self.register_buffer(level + "_suffix", embedded[:, 1 + n_ctx:].clone())
            self.register_buffer(level + "_eot", tokenized.argmax(1))
        self.eval()

    def forward(self, level):
        if level not in ("leaf", "parent"):
            raise ValueError("Invalid prompt hierarchy level")
        prefix, suffix = getattr(self, level + "_prefix"), getattr(self, level + "_suffix")
        context = self.context[level].unsqueeze(0).expand(len(prefix), -1, -1)
        x = torch.cat((prefix, context, suffix), dim=1) + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(len(x), device=x.device), getattr(self, level + "_eot")] @ self.text_projection
        return F.normalize(x.float(), dim=-1)


def _validate_train(train_group, meta):
    if not isinstance(train_group, dict) or not isinstance(meta, dict):
        raise ValueError("Prompt training requires a TRAIN group and taxonomy")
    leaves, parents, mapping = (meta.get(key) for key in ("leaf_names", "parent_names", "leaf_to_parent"))
    for names in (leaves, parents):
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Invalid prompt taxonomy names")
    if (not isinstance(mapping, (list, tuple)) or len(mapping) != len(leaves)
            or any(type(p) is not int or not 0 <= p < len(parents) for p in mapping)):
        raise ValueError("Invalid prompt taxonomy mapping")
    raw = train_group.get("records")
    if (not isinstance(raw, list) or not raw
            or any(row.get("split") != "train" or row.get("status") != "known" for row in raw)):
        raise ValueError("Prompt gradients require nonempty known TRAIN only")
    records = unique_records(raw)
    hashes = [row["image_sha256"] for row in records]
    if hashes != list(train_group.get("image_sha256", [])):
        raise ValueError("Prompt TRAIN feature hash ordering differs")
    for row in records:
        leaf = row.get("true_leaf")
        if (type(leaf) is not int or not 0 <= leaf < len(leaves)
                or type(row.get("true_parent")) is not int or row["true_parent"] != mapping[leaf]):
            raise ValueError("Invalid prompt TRAIN taxonomy labels")
    indices = train_group.get("record_feature_indices")
    if indices is not None and (not isinstance(indices, list) or len(indices) != len(raw)
            or any(type(i) is not int or not 0 <= i < len(hashes)
                   or hashes[i] != row["image_sha256"] for row, i in zip(raw, indices))):
        raise ValueError("Prompt TRAIN alias-to-feature mapping differs")
    try:
        features = torch.as_tensor(train_group["features"]["clip"]).detach().float().cpu()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Prompt training requires cached vanilla CLIP features") from exc
    if (features.ndim != 2 or features.shape[0] != len(records) or features.shape[1] < 1
            or not bool(torch.isfinite(features).all()) or bool((features.norm(dim=1) < 1e-8).any())):
        raise ValueError("Invalid prompt TRAIN CLIP feature cache")
    return records, hashes, features


def _park_source(source):
    devices = []
    for key in ("encoder", "evidence"):
        module = getattr(source, key, None)
        if isinstance(module, nn.Module):
            parameter = next(module.parameters(), None)
            devices.append((module, parameter.device if parameter is not None else torch.device("cpu")))
            module.cpu()
    bank = getattr(source, "bank", None)
    if bank is not None and hasattr(bank, "parent") and hasattr(bank, "to"):
        devices.append((bank, bank.parent.device))
        bank.to("cpu")
    return devices


def fit_prompt(source, train_group, meta, device="cpu", seed=1, loss="coop", options=None):
    """Return a CPU-only prompt artifact and audit after fixed-budget training.

    ``artifact['text_features']`` contains ``leaf`` and ``parent`` unit vectors
    usable directly with cached CLIP features; inference does not reload CLIP.
    """
    settings = _options(options)
    if loss not in ("coop", "fa") or type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("Invalid prompt loss or seed")
    records, hashes, features = _validate_train(train_group, meta)
    if hasattr(source, "meta") and any(source.meta.get(key) != meta.get(key)
                                       for key in ("leaf_names", "parent_names", "leaf_to_parent")):
        raise ValueError("Prompt taxonomy differs from the frozen source")
    feature_digest = tensor_hash(features)
    device = torch.device(device)
    core, model, restore = None, None, []
    try:
        # Build only on CPU, then release source GPU residency before moving
        # the text-only learner. The visual core never enters the GPU.
        core = build_clip_core(source, "cpu")
        before = core_tensor_hash(core)
        if features.shape[1] != core.text_projection.shape[1]:
            raise ValueError("Cached CLIP feature dimension differs from text core")
        reference = {level: encode_templates(core, meta[level + "_names"], SINGLE_TEMPLATES)
                     for level in ("leaf", "parent")}
        scale = float(core.logit_scale.detach().exp().cpu())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Frozen CLIP logit scale is invalid")
        model = ShallowPrompts(core, meta, settings["n_ctx"])
        initial = {key: value.detach().cpu().clone() for key, value in model.context.items()}
        restore = _park_source(source)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.to(device).eval()
        reference = {key: value.to(device).detach() for key, value in reference.items()}
        x_all = F.normalize(features, dim=-1)
        labels = {level: torch.tensor([row["true_" + level] for row in records], dtype=torch.long)
                  for level in ("leaf", "parent")}
        optimizer = torch.optim.SGD(model.context.parameters(), lr=settings["lr"], momentum=.9)
        generator = torch.Generator().manual_seed(seed)
        steps, history, order_digests = 0, [], []
        with torch.no_grad():
            initial_gap = {level: float((model(level) - reference[level]).abs().max())
                           for level in ("leaf", "parent")}
        for epoch in range(1, settings["epochs"] + 1):
            order = torch.randperm(len(records), generator=generator)
            order_digests.append(tensor_hash(order))
            totals, count, gradient_l2_sum = dict(leaf=0., parent=0., loss=0.), 0, 0.
            for batch in order.split(settings["batch_size"]):
                x = x_all[batch].to(device)
                values = {}
                for level in ("leaf", "parent"):
                    logits = scale * x @ model(level).T
                    ref_logits = scale * x @ reference[level].T
                    values[level] = prompt_objective(logits, labels[level][batch].to(device),
                            ref_logits, loss, settings["reference_weight"], settings["temperature"])
                objective = (values["leaf"] + values["parent"]) / 2
                if not bool(torch.isfinite(objective)):
                    raise ValueError("Non-finite TRAIN prompt objective")
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                gradients = [parameter.grad for parameter in model.context.values()]
                if any(value is None or not bool(torch.isfinite(value).all()) for value in gradients):
                    raise ValueError("Missing or non-finite prompt context gradient")
                gradient_l2_sum += math.sqrt(sum(float(value.detach().double().square().sum()) for value in gradients))
                optimizer.step()
                steps += 1
                count += len(batch)
                for key, value in dict(values, loss=objective).items():
                    totals[key] += float(value.detach()) * len(batch)
            history.append(dict(epoch=epoch, optimizer_steps=steps, context_gradient_l2_sum=gradient_l2_sum,
                                **{key: value / count for key, value in totals.items()}))
        with torch.no_grad():
            text_features = {level: model(level).detach().float().cpu().clone() for level in ("leaf", "parent")}
        context = {key: value.detach().float().cpu().clone() for key, value in model.context.items()}
        if any(not bool(torch.isfinite(value).all()) for value in (*text_features.values(), *context.values())):
            raise ValueError("Prompt training produced non-finite output")
        model.cpu()
        core.cpu()
        after = core_tensor_hash(core)
        if before != after:
            raise ValueError("Prompt training modified frozen CLIP weights")
        deltas = {key: float((context[key] - initial[key]).double().norm()) for key in context}
        provenance = dict(source_binding=copy.deepcopy(getattr(source, "binding", {})),
            clip_core_sha256=before, input_feature_sha256=feature_digest, train_image_sha256=hashes,
            train_labels_sha256=object_hash([{key: row[key] for key in
                ("image_sha256", "true_leaf", "true_parent", "split", "status")} for row in records]),
            taxonomy_sha256=object_hash(meta), options=settings, seed=seed,
            initial_context=model.context_text, reference_templates=list(SINGLE_TEMPLATES),
            reference_text_sha256={key: tensor_hash(value) for key, value in reference.items()},
            text_sha256={key: tensor_hash(value) for key, value in text_features.items()},
            batch_order_sha256=object_hash(order_digests), logit_scale=scale,
            scaling="frozen_clip_exp_logit_scale_then_divide_temperature",
            adaptation="hierarchical_shallow_context_cached_images_scaled_FA_denominator",
            context_placement="shallow_input_only", visual_core_used_during_optimization=False)
        artifact = dict(schema_version=SCHEMA_VERSION, loss=loss, context=context,
                        text_features=text_features, meta=copy.deepcopy(meta), provenance=provenance)
        report = dict(schema_version=SCHEMA_VERSION, loss=loss, options=settings, seed=seed,
            gradient_splits=["train"], unknown_images_used_for_gradients=False,
            development_used_for_selection=False, test_used_for_fitting=False,
            selection="final_fixed_budget_epoch", selected_epoch=settings["epochs"],
            optimizer_steps=steps, train_unique_images=len(records), image_sha256=hashes,
            trainable_parameters=sum(value.numel() for value in context.values()),
            trainable_parameter_names=["context.leaf", "context.parent"],
            context_delta_l2=deltas, parameter_delta_l2=math.sqrt(sum(value ** 2 for value in deltas.values())),
            changed_tensor_count=sum(int(value > 0) for value in deltas.values()),
            frozen_core_sha256_before=before, frozen_core_sha256_after=after, frozen_weights_unchanged=True,
            initial_reference_max_abs_gap=initial_gap, batch_order_sha256=provenance["batch_order_sha256"],
            logit_scale=scale, history=history, provenance=copy.deepcopy(provenance))
        return artifact, report
    finally:
        if model is not None:
            model.cpu()
        if core is not None:
            core.cpu()
        del model, core
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for module, original_device in restore:
            module.to(original_device)
