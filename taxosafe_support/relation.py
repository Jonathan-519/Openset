"""Symmetric, coordinate-aware image/reference verification.

The original reference comparator compresses each pair into three cosines.
This opt-in comparator also sees channel-wise differences and products in a
small shared projection. It uses neither class identity nor bank statistics.
Local correspondences use the original normalized token cosine; all nearest
ties share equal weight, and both matching directions receive equal weight.
"""
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class RelationMatcher(nn.Module):
    """One depth-shared verifier with a fixed global-cosine initial prior.

    Feature order is: global cosine, mean/minimum directional local coverage,
    global absolute difference/product, local absolute difference/product.
    Projected vectors are normalized, so feature scale cannot grow by merely
    increasing projection weights. Detached TRAIN vectors still train the
    shared projection on both sides; gradients never enter cached vectors.
    """
    # Bound each expanded coordinate comparison, independent of bank size.
    _coordinate_budget = 262144

    def __init__(self, dimension, relation_dim=32, hidden_dim=32, temperature=.1):
        super().__init__()
        if isinstance(relation_dim, bool) or not isinstance(relation_dim, Integral) or relation_dim < 1:
            raise ValueError("relation_dim must be a positive integer")
        if min(int(dimension), int(relation_dim), int(hidden_dim)) < 1 or float(temperature) <= 0:
            raise ValueError("Invalid relation dimensions/temperature")
        self.dimension, self.relation_dim = int(dimension), int(relation_dim)
        self.projection = nn.Linear(self.dimension, self.relation_dim, bias=False)
        self.residual = nn.Sequential(nn.Linear(3 + 4 * self.relation_dim, int(hidden_dim)),
                                      nn.GELU(), nn.Linear(int(hidden_dim), 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.temperature = float(temperature)

    def _project(self, value):
        return F.normalize(self.projection(value), dim=-1)

    @staticmethod
    def _local_weights(similarities):
        # Equality intentionally includes every exact nearest-neighbour tie.
        # amax also distributes the scalar-coverage gradient over tied tokens.
        qmax = similarities.amax(-1, keepdim=True)
        rmax = similarities.amax(-2, keepdim=True)
        forward = (similarities == qmax).to(similarities.dtype)
        backward = (similarities == rmax).to(similarities.dtype)
        forward = forward / forward.sum(-1, keepdim=True)
        backward = backward / backward.sum(-2, keepdim=True)
        weight = .5 * (forward / similarities.shape[-2] + backward / similarities.shape[-1])
        qcoverage, rcoverage = qmax.squeeze(-1).mean(-1), rmax.squeeze(-2).mean(-1)
        return weight, (qcoverage + rcoverage) / 2, torch.minimum(qcoverage, rcoverage)

    def pair_features(self, query, references, query_local=None, reference_local=None):
        """Return [B,N,3+4r] features with bounded temporary pair tensors."""
        query = F.normalize(query.float(), dim=-1)
        references = F.normalize(references.detach().float(), dim=-1)
        if query.ndim != 2 or references.ndim != 2 or query.shape[-1] != self.dimension or references.shape[-1] != self.dimension:
            raise ValueError("Relation vectors must have shapes [B,D] and [N,D]")
        qprojected, rprojected = self._project(query), self._project(references)
        if not len(query) or not len(references):
            # Physical banks are nonempty in SupportBank. Keep the public
            # comparator well-defined for shape-only and isolated empty calls.
            return query.new_zeros((len(query), len(references), 3 + 4 * self.relation_dim)) + (qprojected.sum() + rprojected.sum()) * 0
        use_local = query_local is not None and reference_local is not None
        if use_local:
            query_local = F.normalize(query_local.float(), dim=-1)
            reference_local = F.normalize(reference_local.detach().float(), dim=-1)
            if (query_local.ndim != 3 or reference_local.ndim != 3
                    or query_local.shape[0] != len(query) or reference_local.shape[0] != len(references)
                    or query_local.shape[-1] != self.dimension or reference_local.shape[-1] != self.dimension
                    or min(query_local.shape[1], reference_local.shape[1]) < 1):
                raise ValueError("Relation local vectors must have nonempty [B,K,D] and [N,J,D] shapes")
            qlocal, rlocal = self._project(query_local), self._project(reference_local)
            per_pair = query_local.shape[1] * reference_local.shape[1] * self.relation_dim
        else:
            per_pair = self.relation_dim
        # Chunk both query and reference dimensions; no per-image Python loop,
        # no data-dependent device synchronization, and one projection per side.
        pair_limit = max(1, self._coordinate_budget // per_pair)
        query_chunk = min(max(1, len(query)), pair_limit)
        batches = []
        for begin in range(0, len(query), query_chunk):
            end = min(begin + query_chunk, len(query))
            reference_chunk = max(1, pair_limit // (end - begin))
            rows = []
            for start in range(0, len(references), reference_chunk):
                stop = min(start + reference_chunk, len(references))
                global_similarity = query[begin:end] @ references[start:stop].T
                q, r = qprojected[begin:end, None, :], rprojected[None, start:stop, :]
                difference, product = (q - r).abs(), q * r
                if use_local:
                    similarities = torch.einsum("bkd,njd->bnkj", query_local[begin:end], reference_local[start:stop])
                    weight, mean, minimum = self._local_weights(similarities)
                    ql = qlocal[begin:end, None, :, None, :]
                    rl = rlocal[None, start:stop, None, :, :]
                    local_difference = ((ql - rl).abs() * weight[..., None]).sum((-3, -2))
                    local_product = ((ql * rl) * weight[..., None]).sum((-3, -2))
                else:
                    mean = minimum = global_similarity
                    local_difference = torch.zeros_like(difference)
                    local_product = torch.zeros_like(product)
                scalars = torch.stack((global_similarity, mean, minimum), -1)
                rows.append(torch.cat((scalars, difference, product, local_difference, local_product), -1))
            batches.append(torch.cat(rows, 1))
        return torch.cat(batches, 0)

    def forward(self, query, references, query_local=None, reference_local=None):
        features = self.pair_features(query, references, query_local, reference_local)
        return (features[..., 0] - .5) / self.temperature + self.residual(features).squeeze(-1)
