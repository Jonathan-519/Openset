"""Class-specific reconstruction on detached, known-only feature caches.

This is a frozen-feature adaptation inspired by CSSR, not a reproduction of
its jointly trained backbone. Each class has a bias-free D->rank->D tanh
autoencoder. Classification averages per-position probabilities, as in the
official CSSR ``softmax_avg`` criterion. The final rejection score evaluates
only the caller's existing candidate; it never changes identity or parent rank.
"""
import hashlib
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(value)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def _labels(labels, length, classes, device):
    original = torch.as_tensor(labels, device=device)
    if original.shape != (length,) or original.dtype == torch.bool or original.is_complex():
        raise ValueError("One integer class label per image is required")
    if original.is_floating_point() and (not bool(torch.isfinite(original).all()) or not bool((original == original.round()).all())):
        raise ValueError("Class labels must be finite integers")
    result = original.to(torch.long)
    if bool(((result < 0) | (result >= classes)).any()):
        raise ValueError("Class label is outside the reconstruction taxonomy")
    return result


def _active(active_mask, classes, device="cpu"):
    if active_mask is None:
        return torch.ones(classes, dtype=torch.bool, device=device)
    result = torch.as_tensor(active_mask, dtype=torch.bool, device=device)
    if result.shape != (classes,) or not bool(result.any()):
        raise ValueError("active_mask must select at least one class")
    return result.clone()


class ClassSpecificReconstruction(nn.Module):
    """Small independent class autoencoders; no image encoder is owned here.

    ``relative`` uses -L1 reconstruction error / L1 feature magnitude, suitable
    for the already L2-normalized fine embedding. ``cssr`` divides by squared
    L1 magnitude, a separate score ablation. CSSR's public implementation uses
    squared *mean* absolute magnitude and a fixed gamma; at fixed D these
    differ by a constant scale. Here the positive classification scale is
    learned, and is deliberately excluded from the absolute rejection score.
    """
    def __init__(self, dimension, num_classes, rank=16, active_mask=None,
                 scale_init=1., score_mode="relative", eps=1e-8):
        super().__init__()
        self.dimension = _integer(dimension, "dimension")
        self.num_classes = _integer(num_classes, "num_classes")
        self.rank = _integer(rank, "rank")
        self.eps = _positive(eps, "eps")
        if not math.sqrt(torch.finfo(torch.float32).tiny) <= self.eps < 1:
            raise ValueError("eps must have a representable float32 square and be smaller than one")
        if score_mode not in ("relative", "cssr"):
            raise ValueError("score_mode must be relative or cssr")
        self.score_mode = score_mode
        scale_init = _positive(scale_init, "scale_init")
        self.scale_init = scale_init
        # Stable inverse softplus without exp(large_scale).
        raw_scale = scale_init + math.log(-math.expm1(-scale_init))
        self.raw_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))
        if not bool(torch.isfinite(self.raw_scale)):
            raise ValueError("scale_init must be representable in float32")
        self.class_aes = nn.ModuleList([
            nn.Sequential(nn.Linear(self.dimension, self.rank, bias=False), nn.Tanh(),
                          nn.Linear(self.rank, self.dimension, bias=False))
            for _ in range(self.num_classes)])
        self.register_buffer("active_mask", _active(active_mask, self.num_classes))

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + torch.finfo(self.raw_scale.dtype).tiny

    def _features(self, features):
        features = torch.as_tensor(features, device=self.raw_scale.device).float()
        if features.ndim == 2:
            features = features[:, None, :]
        if features.ndim != 3 or features.shape[-1] != self.dimension or features.shape[1] < 1:
            raise ValueError("Features must have shape [B,D] or nonempty [B,P,D]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Reconstruction features must be finite")
        return features

    def errors(self, features):
        """Return unscaled, nonnegative L1 errors [B,P,C].

        Classes are evaluated one at a time to avoid materializing [B,P,C,D].
        Inactive classes are masked by the classifier and scorer, not relabeled.
        """
        features = self._features(features)
        return torch.stack([(autoencoder(features) - features).abs().sum(-1)
                            for autoencoder in self.class_aes], -1)

    def log_probabilities(self, features):
        errors = self.errors(features)
        logits = (-errors * self.scale).masked_fill(~self.active_mask, -torch.inf)
        per_position = F.log_softmax(logits, dim=-1)
        # log(mean(softmax(logits))) is stable even for a difficult class.
        # Never differentiate logsumexp(-inf,...,-inf) for an inactive class:
        # replace its temporary values with finite zeros, then mask it again.
        safe = per_position.masked_fill(~self.active_mask, 0.)
        pooled = torch.logsumexp(safe, dim=1) - math.log(errors.shape[1])
        return pooled.masked_fill(~self.active_mask, -torch.inf)

    def forward(self, features):
        return self.log_probabilities(features)

    def loss(self, features, labels):
        probabilities = self.log_probabilities(features)
        labels = _labels(labels, len(probabilities), self.num_classes, probabilities.device)
        if not len(labels) or not bool(self.active_mask[labels].all()):
            raise ValueError("Training labels must be nonempty and active")
        return F.nll_loss(probabilities, labels)

    def candidate_scores(self, features, candidate_leaf):
        """Higher is more known; retain the baseline candidate exactly.

        A zero/near-zero position or inactive candidate fails closed to the
        finite dtype minimum. Such a feature must never receive score zero
        merely because both its input and bias-free reconstruction are zero.
        """
        features = self._features(features)
        candidate = _labels(candidate_leaf, len(features), self.num_classes, features.device)
        errors = self.errors(features)
        selected = errors.gather(-1, candidate[:, None, None].expand(-1, features.shape[1], 1)).squeeze(-1)
        magnitude = features.abs().sum(-1)
        denominator = magnitude.clamp_min(self.eps)
        if self.score_mode == "cssr":
            denominator = denominator.square()
        per_position = -selected / denominator
        score = per_position.mean(-1)
        valid = ((magnitude > self.eps).all(-1) & torch.isfinite(magnitude).all(-1) & self.active_mask[candidate]
                 & torch.isfinite(per_position).all(-1) & torch.isfinite(score))
        return torch.where(valid, score, torch.full_like(score, torch.finfo(score.dtype).min))

    def constructor_arguments(self):
        """JSON-compatible metadata to accompany an ordinary state_dict."""
        return {"dimension": self.dimension, "num_classes": self.num_classes, "rank": self.rank,
                "active_mask": self.active_mask.detach().cpu().tolist(), "scale_init": self.scale_init,
                "score_mode": self.score_mode, "eps": self.eps}


def stratified_hash_split(labels, content_hashes, num_classes, *, active_mask=None,
                          validation_fraction=.2, seed=1):
    """Split unique known TRAIN hashes without using validation/unknown rows.

    Singleton classes remain in fitting. Every class with >=2 unique images
    contributes to the inner image holdout. This is auxiliary early stopping,
    never a strict unseen-class or backbone-independent validation experiment.
    """
    num_classes = _integer(num_classes, "num_classes")
    seed = _integer(seed, "seed", minimum=0)
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    labels = _labels(labels, len(content_hashes), num_classes, "cpu")
    active = _active(active_mask, num_classes)
    if not len(labels) or not bool(active[labels].all()):
        raise ValueError("Cached labels must be nonempty and active")
    unique = {}
    for row, (content_hash, label) in enumerate(zip(content_hashes, labels.tolist())):
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("Every cached image requires a nonempty content hash")
        if content_hash in unique and int(labels[unique[content_hash]]) != label:
            raise ValueError("A content hash has conflicting class labels")
        unique.setdefault(content_hash, row)
    fitting, validation, classes = [], [], []
    for label in range(num_classes):
        if not bool(active[label]):
            continue
        hashes = [h for h, row in unique.items() if int(labels[row]) == label]
        if not hashes:
            raise ValueError("Missing known TRAIN features for active class %d" % label)
        hashes.sort(key=lambda h: (hashlib.sha256((str(seed) + "\0" + h).encode("utf-8")).hexdigest(), h))
        count = min(len(hashes) - 1, max(1, int(len(hashes) * fraction))) if len(hashes) > 1 else 0
        validation.extend(unique[h] for h in hashes[:count])
        fitting.extend(unique[h] for h in hashes[count:])
        classes.append({"class_id": label, "unique_images": len(hashes),
                        "fit_images": len(hashes) - count, "validation_images": count})
    if not validation:
        raise ValueError("Known TRAIN cache cannot form a nonempty inner image validation split")
    return {"source": "known_train_inner_images", "strict_unseen_class_evaluation": False,
            "frozen_backbone_independent_validation": False,
            "seed": seed, "validation_fraction": fraction,
            "fit_indices": fitting, "validation_indices": validation,
            "fit_hashes": [content_hashes[i] for i in fitting],
            "validation_hashes": [content_hashes[i] for i in validation],
            "duplicate_rows_removed": len(labels) - len(unique), "per_class": classes}


def fit_cached_features(features, labels, content_hashes, num_classes, *, rank=16,
                        active_mask=None, epochs=40, batch_size=128, learning_rate=1e-3,
                        validation_fraction=.2, patience=8, seed=1, device="cpu",
                        score_mode="relative", scale_init=1., eps=1e-8):
    """Fit only autoencoders/scale; return (best_model, JSON training report).

    The caller must prove these are known TRAIN features from a frozen encoder.
    This function accepts no unknown or external validation features. Duplicate
    hashes are deduplicated only when their labels AND exact feature bytes agree.
    Cache order cannot change the split, minibatches, initialization or fit.
    """
    epochs = _integer(epochs, "epochs")
    batch_size = _integer(batch_size, "batch_size")
    patience = _integer(patience, "patience")
    seed = _integer(seed, "seed", minimum=0)
    learning_rate = _positive(learning_rate, "learning_rate")
    features = torch.as_tensor(features).detach().cpu().float().clone()
    if features.ndim not in (2, 3) or len(features) != len(content_hashes) or not len(features):
        raise ValueError("Cached features require one [D] or [P,D] tensor per content hash")
    if features.shape[-1] < 1 or (features.ndim == 3 and features.shape[1] < 1) or not bool(torch.isfinite(features).all()):
        raise ValueError("Cached features must be nonempty and finite")
    num_classes = _integer(num_classes, "num_classes")
    labels = _labels(labels, len(features), num_classes, "cpu")
    split = stratified_hash_split(labels, content_hashes, num_classes, active_mask=active_mask,
                                   validation_fraction=validation_fraction, seed=seed)
    first = {}
    for row, content_hash in enumerate(content_hashes):
        if content_hash in first and not torch.equal(features[first[content_hash]], features[row]):
            raise ValueError("A content hash has conflicting cached features")
        first.setdefault(content_hash, row)
    device = torch.device(device)
    # Initialize on CPU under a local RNG scope; do not reseed the parent
    # pipeline or touch CUDA RNG state. Batch permutations own a CPU generator.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ClassSpecificReconstruction(features.shape[-1], num_classes, rank, active_mask,
                                            scale_init, score_mode, eps).to(device)
    if int(model.active_mask.sum()) < 2:
        raise ValueError("Reconstruction classification fitting requires at least two active classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fitting = torch.tensor(split["fit_indices"], dtype=torch.long)
    validation = torch.tensor(split["validation_indices"], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def evaluate(indices):
        model.eval()
        total, correct = 0., 0
        for block in indices.split(batch_size):
            log_probabilities = model.log_probabilities(features[block].to(device))
            target = labels[block].to(device)
            total += float(F.nll_loss(log_probabilities, target, reduction="sum").cpu())
            correct += int((log_probabilities.argmax(-1) == target).sum().cpu())
        return total / len(indices), correct / len(indices)

    initial_loss, initial_accuracy = evaluate(validation)
    if not math.isfinite(initial_loss):
        raise RuntimeError("Initial reconstruction validation loss is not finite")
    best_loss, best_epoch = initial_loss, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = fitting[torch.randperm(len(fitting), generator=generator)]
        total = 0.
        for block in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(features[block].to(device), labels[block].to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Reconstruction training loss became nonfinite")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise RuntimeError("Reconstruction training gradient became nonfinite")
            optimizer.step()
            total += float(loss.detach().cpu()) * len(block)
        validation_loss, validation_accuracy = evaluate(validation)
        if not math.isfinite(validation_loss):
            raise RuntimeError("Reconstruction validation loss became nonfinite")
        improved = validation_loss < best_loss
        history.append({"epoch": epoch, "fit_nll": total / len(fitting),
                        "validation_nll": validation_loss, "validation_accuracy": validation_accuracy,
                        "classification_scale": float(model.scale.detach().cpu()), "selected": improved})
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    report = {"schema": "taxosafe_refine_fit_v1", "feature_shape": list(features.shape),
              "feature_cache_detached": True, "split": split,
              "selection_metric": "known_train_inner_images_mean_nll",
              "initial_validation_nll": initial_loss, "initial_validation_accuracy": initial_accuracy,
              "best_epoch": best_epoch, "best_validation_nll": best_loss,
              "epochs_requested": epochs, "epochs_completed": len(history),
              "early_stopped": len(history) < epochs, "patience": patience,
              "batch_size": batch_size, "learning_rate": learning_rate, "optimizer": "Adam",
              "seed": seed, "model_arguments": model.constructor_arguments(), "history": history}
    return model, report
