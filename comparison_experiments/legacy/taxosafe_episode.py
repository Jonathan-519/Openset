"""Deterministic episode helpers shared by TaxoSafe samplers.

The training DataLoader and the open-treecut generator must agree on which
known leaf is temporarily treated as pseudo-unseen.  Keeping this tiny helper
in one module prevents the two components from silently drifting apart.
"""


def epoch_holdout_leaf(children, parent_id, epoch, seed):
    """Return the leaf hidden for one parent during an entire epoch.

    The offset is parent specific and the epoch advances it by one position,
    so every leaf is used as pseudo-unseen before the schedule repeats.
    Singleton branches return ``None`` because they have no active sibling
    against which novelty can be learned.
    """
    ordered = sorted(int(value) for value in children)
    if len(ordered) < 2:
        return None

    offset = (int(seed) + 104729 * int(parent_id)) % len(ordered)
    index = (offset + int(epoch)) % len(ordered)
    return ordered[index]
