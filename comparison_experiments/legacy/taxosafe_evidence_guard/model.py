"""Zero-initialized query adaptation around the exact frozen C00 verifier.

All four global/local query views and the original reference bank remain in
the inference path. No new ranking or membership head replaces a C00 head.
"""
import copy

import torch
from torch import nn

from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.support import SupportBank


RAW_FIELDS = ("parent_logits", "leaf_logits", "parent_membership_logits", "leaf_membership_logits")


def restore_bank(state, device="cpu"):
    """Validate through the original constructor, then preserve saved bits.

    SupportBank.from_state_dict normalizes already normalized saved vectors.
    Restoring the validated original tensors avoids that second roundoff.
    """
    state = copy.deepcopy(state)
    bank = SupportBank.from_state_dict(state)
    for name, value in state.items():
        setattr(bank, name, value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value))
    return bank.to(device)


class QueryResidual(nn.Module):
    def __init__(self, dimension, hidden, bound):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.bound = float(bound)

    def forward(self, value):
        # Each query/token residual has a bounded L2 norm. Zero initialization
        # returns precisely the original tensor values before verifier logic.
        delta = torch.tanh(self.net(value.float()))
        delta = delta * (self.bound / max(1., float(value.shape[-1]) ** .5))
        return value.float() + delta


class EvidenceGuard(nn.Module):
    def __init__(self, context, arm, settings, geometry_settings=None, geometry_state=None):
        super().__init__()
        self.arm = copy.deepcopy(arm)
        self.settings = copy.deepcopy(settings)
        self.geometry_settings = copy.deepcopy(geometry_settings or {})
        self.geometry_state = geometry_state
        self.meta = copy.deepcopy(context["meta"])
        dimension = int(context["dimension"])
        self.verifier = HierarchicalEvidence(dimension, self.meta["leaf_to_parent"],
                                            **copy.deepcopy(context["evidence_kwargs"]))
        self.verifier.load_state_dict(context["evidence_state"], strict=True)
        if not self.verifier.decoupled or self.verifier.membership_mode != "reference":
            raise ValueError("EvidenceGuard requires the original decoupled C00 reference verifier")
        self.verifier.requires_grad_(False).eval()
        self.bank = restore_bank(context["bank_state"])
        if self.bank.leaf_to_parent.tolist() != self.meta["leaf_to_parent"]:
            raise ValueError("C00 bank and metadata taxonomy differ")
        self.adapters = nn.ModuleDict()
        for branch in ("parent", "fine"):
            if arm.get("adapt_" + branch, False):
                self.adapters[branch] = QueryResidual(dimension, int(settings["adapter_dim"]),
                                                      float(settings["feature_bound"]))
        self.geometry_heads = nn.ModuleDict()
        if arm.get("geometry", False):
            if geometry_state is None:
                raise ValueError("Geometry arm requires known-TRAIN geometry state")
            for branch in ("parent", "leaf"):
                head = nn.Sequential(nn.Linear(6, int(self.geometry_settings.get("hidden", 16))),
                                     nn.GELU(), nn.Linear(int(self.geometry_settings.get("hidden", 16)), 1))
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.geometry_heads[branch] = head

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.bank.to(next(self.verifier.parameters()).device)
        return result

    def train(self, mode=True):
        super().train(mode)
        self.verifier.eval()
        return self

    def adapt(self, encoded):
        result = dict(encoded)
        norms = []
        for branch, adapter in self.adapters.items():
            for key in (branch, branch + "_local"):
                if encoded.get(key) is None:
                    continue
                result[key] = adapter(encoded[key])
                delta = (result[key] - encoded[key].float()).square().sum(-1)
                if delta.ndim > 1:
                    delta = delta.mean(-1)
                norms.append(delta)
        norm = torch.stack(norms).mean(0) if norms else encoded["fine"].new_zeros(len(encoded["fine"]))
        return result, norm

    def teacher(self, encoded, query_hashes=None):
        return self.verifier(encoded, self.bank, query_hashes=query_hashes)

    def _replace_membership(self, output, parent_membership, leaf_membership):
        stats = {"parent_active": output["active_parents"], "leaf_active": output["active_leaves"]}
        rebuilt = self.verifier._decoupled_output(output["parent_features"], output["leaf_features"],
            output["parent_logits"], output["leaf_logits"], stats,
            membership_logits=(parent_membership, leaf_membership))
        return dict(output, **rebuilt)

    def centered(self, output, router):
        """Loss probabilities are centered on a frozen reference operating point."""
        return self._replace_membership(output,
            output["parent_membership_logits"] - float(router["parent_threshold"]),
            output["leaf_membership_logits"] - float(router["leaf_threshold"]))

    def forward(self, encoded, query_hashes=None, geometry=None):
        adapted, residual_norm = self.adapt(encoded)
        output = self.verifier(adapted, self.bank, query_hashes=query_hashes)
        if self.geometry_heads:
            if geometry is None:
                from .geometry import score
                with torch.no_grad():
                    geometry = score(encoded, query_hashes, self.geometry_state,
                                     device=encoded["fine"].device)
            changes = {}
            bound = float(self.geometry_settings.get("residual_bound", 2.))
            for branch, head in self.geometry_heads.items():
                features = geometry[branch].to(encoded["fine"].device).detach()
                if features.ndim != 3 or features.shape[-1] != 6 or not torch.isfinite(features).all():
                    raise ValueError("Invalid finite six-dimensional geometry features")
                changes[branch] = bound * torch.tanh(head(features.float()).squeeze(-1))
            output = self._replace_membership(output,
                output["parent_membership_logits"] + changes["parent"],
                output["leaf_membership_logits"] + changes["leaf"])
        output["guard_residual_norm"] = residual_norm
        return output
