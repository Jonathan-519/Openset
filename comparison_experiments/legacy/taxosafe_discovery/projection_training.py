"""Fixed-budget TRAIN-only feature adapter: text CE and supervised contrast."""
import math

import torch
from torch.nn import functional as F

from taxosafe_support.calibration import unique_records
from .features import tensor_hash
from .models import ResidualProjection


DEFAULTS = {"epochs": 20, "batch_size": 64, "learning_rate": .001,
            "bottleneck": 64, "temperature": .1, "supcon_weight": .1}


def _options(options):
    result = dict(DEFAULTS)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(result):
            raise ValueError("Unknown projection training options")
        result.update(options)
    for key in ("epochs", "batch_size", "bottleneck"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Projection " + key + " must be a positive integer")
    for key in ("learning_rate", "temperature", "supcon_weight"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid projection " + key)
        if value < 0 or (key != "supcon_weight" and value == 0):
            raise ValueError("Invalid projection " + key)
    return result


def supervised_contrastive(features, labels, hashes, temperature=.1):
    """Only same-class different-content positives; singleton anchors are skipped."""
    if len(features) != len(labels) or len(hashes) != len(features):
        raise ValueError("Contrastive batch alignment differs")
    logits = F.normalize(features, dim=-1) @ F.normalize(features, dim=-1).T / temperature
    different = torch.tensor([[a != b for b in hashes] for a in hashes],
                             dtype=torch.bool, device=features.device)
    positive = (labels[:, None] == labels[None, :]) & different
    valid = positive.any(1) & different.any(1)
    if not bool(valid.any()):
        return features.sum() * 0, 0
    logits = logits[valid]
    allowed = different[valid]
    positive = positive[valid]
    log_denominator = logits.masked_fill(~allowed, -torch.inf).logsumexp(1)
    # where avoids 0 * -inf and computes the ordinary per-positive mean.
    terms = torch.where(positive, logits - log_denominator[:, None], 0.)
    loss = -(terms.sum(1) / positive.sum(1)).mean()
    return loss, int(valid.sum())


def fit_projection(train_group, ensemble_leaf, meta, device="cpu", seed=1, options=None):
    """Fit only supplied known TRAIN, always return the final fixed-budget state.

    No development argument or early stopping exists. The text vectors and
    cached image features are detached, and only this small adapter has grads.
    """
    settings = _options(options)
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("Invalid projection seed")
    raw = list(train_group.get("records", []))
    if not raw or any(row.get("split") != "train" or row.get("status") != "known" for row in raw):
        raise ValueError("Projection gradients require nonempty known TRAIN only")
    records = unique_records(raw)
    hashes = [row["image_sha256"] for row in records]
    if hashes != list(train_group.get("image_sha256", [])):
        raise ValueError("TRAIN feature hash ordering differs from unique records")
    mapping = meta["leaf_to_parent"]
    for row in records:
        label = row.get("true_leaf")
        if (type(label) is not int or not 0 <= label < len(mapping)
                or row.get("true_parent") != mapping[label]):
            raise ValueError("Invalid projection TRAIN taxonomy")
    features = torch.as_tensor(train_group["features"]["clip"]).detach().float().cpu()
    text = torch.as_tensor(ensemble_leaf).detach().float().cpu()
    if (features.ndim != 2 or len(features) != len(records)
            or text.shape != (len(mapping), features.shape[1])
            or not bool(torch.isfinite(features).all() and torch.isfinite(text).all())
            or bool((features.norm(dim=-1) < 1e-8).any()) or bool((text.norm(dim=-1) < 1e-8).any())):
        raise ValueError("Invalid projection feature/text cache")
    input_feature_sha256, input_text_sha256 = tensor_hash(features), tensor_hash(text)
    device = torch.device(device)
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResidualProjection(features.shape[1], settings["bottleneck"])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"], momentum=.9)
    features = F.normalize(features, dim=-1)
    text = F.normalize(text.to(device), dim=-1)
    labels = torch.tensor([row["true_leaf"] for row in records], dtype=torch.long)
    history, steps = [], 0
    for epoch in range(1, settings["epochs"] + 1):
        sums = {"loss": 0., "cross_entropy": 0., "supervised_contrastive": 0.}
        count, contrastive_anchors = 0, 0
        order = torch.randperm(len(features), generator=generator)
        for batch in order.split(settings["batch_size"]):
            x, y = features[batch].to(device), labels[batch].to(device)
            projected = model(x)
            ce = F.cross_entropy(projected @ text.T / settings["temperature"], y)
            contrast, valid = supervised_contrastive(projected, y,
                    [hashes[i] for i in batch.tolist()], settings["temperature"])
            loss = ce + settings["supcon_weight"] * contrast
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite TRAIN projection objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            count += len(batch)
            contrastive_anchors += valid
            for key, value in (("loss", loss), ("cross_entropy", ce), ("supervised_contrastive", contrast)):
                sums[key] += float(value.detach()) * len(batch)
        history.append({"epoch": epoch, "optimizer_steps": steps,
                        **{key: value / count for key, value in sums.items()},
                        "valid_supcon_anchors": contrastive_anchors,
                        "skipped_supcon_anchors": count - contrastive_anchors})
    state = model.cpu().state_dict()
    optimizer.zero_grad(set_to_none=True)
    deltas = [state[key] - value for key, value in initial.items()]
    squared = sum(float(value.double().square().sum()) for value in deltas)
    if not math.isfinite(squared):
        raise ValueError("Projection produced non-finite parameter updates")
    report = {"schema_version": "discovery_projection_v1", "options": settings, "seed": seed,
              "gradient_splits": ["train"], "unknown_images_used_for_gradients": False,
              "development_used_for_selection": False, "test_used_for_fitting": False,
              "selection": "final_fixed_budget_epoch", "selected_epoch": settings["epochs"],
              "optimizer_steps": steps, "train_unique_images": len(records), "image_sha256": hashes,
              "feature_sha256": tensor_hash(features), "text_sha256": tensor_hash(text),
              "input_feature_sha256": input_feature_sha256, "input_text_sha256": input_text_sha256,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "parameter_delta_l2": math.sqrt(squared),
              "changed_tensor_count": sum(int(bool(value.ne(0).any())) for value in deltas),
              "history": history, "dimension": features.shape[1], "bottleneck": settings["bottleneck"],
              "loss": "fixed_ensemble_text_cross_entropy_plus_same_leaf_distinct_hash_supervised_contrastive"}
    return model.eval().requires_grad_(False), report


@torch.no_grad()
def transform_projection(model, features, batch_size=256):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Projection batch_size must be positive")
    features = torch.as_tensor(features).detach().float()
    if features.ndim != 2 or not len(features) or not bool(torch.isfinite(features).all()):
        raise ValueError("Projection inference requires finite nonempty [N,D] features")
    device = next(model.parameters()).device
    result = torch.cat([model(batch.to(device)).cpu() for batch in features.split(batch_size)])
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Projection inference produced non-finite features")
    return result
