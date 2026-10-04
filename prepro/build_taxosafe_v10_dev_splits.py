"""Build leakage-free TaxoSafe-v10 development train/calibration manifests.

The original v9 development partitions are split *within each species/source*:
60% for gradient supervision (near unknown or OE), 40% held out for router
calibration. Locked test manifests are never read or written.
"""

import argparse
from collections import defaultdict
from pathlib import Path


def _group_key(line, mode):
    path = line.split(",", 1)[0]
    parts = path.split("/")
    if mode == "intra":
        if len(parts) < 2:
            raise ValueError("Near-unknown path lacks parent/species: " + path)
        return "/".join(parts[:2])
    if mode == "extra":
        return parts[0]
    raise ValueError("mode must be intra or extra")


def stratified_split(lines, mode, train_fraction=0.60):
    groups = defaultdict(list)
    for line in lines:
        line = line.strip()
        if line:
            groups[_group_key(line, mode)].append(line)

    train, calibration = [], []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) < 2:
            raise ValueError("Need at least two rows in group {}".format(key))
        n_train = max(1, min(len(rows) - 1, int(len(rows) * train_fraction)))
        # Deterministic permutation independent of filesystem traversal order.
        order = sorted(
            range(len(rows)),
            key=lambda index: ((index * 37) % len(rows), index),
        )
        selected = set(order[:n_train])
        for index, row in enumerate(rows):
            (train if index in selected else calibration).append(row)
    return train, calibration


def _read(path):
    return [
        line for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="prepro/data/Zooplankton_TT_v9_rebuild"
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must lie strictly between 0 and 1")

    root = Path(args.root)
    intra_train, intra_val = stratified_split(
        _read(root / "gt_val_intra.txt"), "intra", args.train_fraction
    )
    extra_train, extra_val = stratified_split(
        _read(root / "gt_val_extra.txt"), "extra", args.train_fraction
    )

    outputs = {
        "gt_train_intra_v10.txt": intra_train,
        "gt_val_intra_v10.txt": intra_val,
        "gt_oe_train_v10.txt": extra_train,
        "gt_val_extra_v10.txt": extra_val,
    }
    for name, rows in outputs.items():
        _write(root / name, rows)
        print("{}: {}".format(name, len(rows)))

    if set(intra_train) & set(intra_val):
        raise RuntimeError("Near-unknown train/calibration overlap detected")
    if set(extra_train) & set(extra_val):
        raise RuntimeError("Global-OOD train/calibration overlap detected")


if __name__ == "__main__":
    main()
