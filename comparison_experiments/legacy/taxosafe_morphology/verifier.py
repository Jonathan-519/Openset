"""Trainable spatial adapters and signed zero-initialized D05 corrections."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .matching import DESCRIPTORS, pair_evidence

SCHEMA_VERSION = "morphology_spatial_verifier_v1"


class SpatialVerifier(nn.Module):
    def __init__(self, dimension, leaves, adapter_dim=32, hidden=64):
        super().__init__()
        self.spec = dict(dimension=dimension, leaves=leaves, adapter_dim=adapter_dim, hidden=hidden)
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, adapter_dim)
        self.up = nn.Linear(adapter_dim, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.delta = nn.Sequential(nn.Linear(2 * dimension + len(DESCRIPTORS), hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.local_classifier = nn.Linear(dimension, leaves)

    def adapt(self, tokens):
        return F.normalize(tokens.float() + self.up(F.gelu(self.down(self.norm(tokens.float())))), dim=-1)

    def evidence(self, query, references, reference_leaves, positions, settings):
        if query.ndim != 2 or references.ndim != 3 or len(references) != len(reference_leaves) or not len(references):
            raise ValueError("Verifier requires one query and at least one real TRAIN reference")
        q = self.adapt(query[None])
        r = self.adapt(references)
        vectors, descriptions, details = [], [], []
        chunk = settings["pair_chunk"]
        for start in range(0, len(r), chunk):
            ref = r[start:start + chunk]
            v, d, detail = pair_evidence(q.expand(len(ref), -1, -1), ref, positions, settings)
            vectors.append(v); descriptions.append(d); details.append(detail)
        vectors, descriptions = torch.cat(vectors), torch.cat(descriptions)
        # Choose a whole real individual, then its child mode. No maximum over
        # per-patch scores from different animals. Every child gets its own
        # reference comparison; only the supported mode need agree.
        modes = []
        labels = torch.as_tensor(reference_leaves, device=r.device)
        for leaf in sorted(set(labels.detach().cpu().tolist())):
            ids = (labels == leaf).nonzero(as_tuple=True)[0]
            winner = ids[descriptions[ids, 0].detach().argmax()]
            modes.append(winner)
        modes = torch.stack(modes)
        chosen = modes[descriptions[modes, 0].detach().argmax()]
        # Same-species additional real references provide a second complete
        # match. Averaging descriptors is after correspondence, never before.
        siblings = (labels == labels[chosen]).nonzero(as_tuple=True)[0]
        order = descriptions[siblings, 0].detach().argsort(descending=True)
        selected = siblings[order[:min(2, len(siblings))]]
        aggregate = vectors[selected].mean(0)
        delta = self.delta(aggregate).squeeze(-1)
        all_details = {key: torch.cat([d[key] for d in details]) for key in details[0]}
        evidence = dict(descriptors=descriptions, selected_reference=int(chosen.detach().cpu()),
                        selected_mode=int(labels[chosen].detach().cpu()),
                        aggregate_references=selected.detach().cpu().tolist(), **all_details)
        return delta, q.mean(1)[0], evidence

    def export_state(self):
        return dict(schema_version=SCHEMA_VERSION, spec=copy.deepcopy(self.spec),
                    tensors={key: value.detach().cpu().clone() for key, value in self.state_dict().items()})

    @classmethod
    def restore(cls, state, device="cpu"):
        if not isinstance(state, dict) or set(state) != {"schema_version", "spec", "tensors"} or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unexpected spatial verifier state")
        with torch.random.fork_rng(devices=[]):
            model = cls(**state["spec"])
        model.load_state_dict(state["tensors"], strict=True)
        if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
            raise ValueError("Nonfinite spatial verifier state")
        return model.to(device).eval().requires_grad_(False)
