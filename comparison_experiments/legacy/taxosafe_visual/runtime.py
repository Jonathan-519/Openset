"""Read-only MaPLe extraction and artifact checks for visual-support v4.

Torch is imported only by GPU extraction. Core and regression tests need only
NumPy/PyYAML/scikit-learn. No training/validation loader is built at test time.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PARENTS = ["Amphipoda", "Appendiculata", "Cladocera", "Copepoda",
                    "Euphausiacea", "Medusae", "Sagittoidea"]
SPLITS = {
    "train": "known", "val_known": "known", "val_intra": "intra",
    "val_extra": "extra", "test_known": "known", "test_intra": "intra",
    "test_extra": "extra",
}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_records(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_configuration(path, trial, run_dir=None):
    cfg_path = resolve(path)
    with cfg_path.open("r", encoding="utf-8-sig") as stream:
        extension = yaml.safe_load(stream)
    if not isinstance(extension, dict) or "base_config" not in extension:
        raise ValueError("Use the provided TaxoSafe_visual.yml with base_config")
    with resolve(extension["base_config"]).open("r", encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)
    cfg["visual_support"] = extension.get("visual_support", {})
    cfg["visual_support"].setdefault("primary_profile", "coverage")
    primary = cfg["visual_support"]["primary_profile"]
    residual = cfg["visual_support"].get("residual", {})
    allowed = ("risk", "coverage", "balanced") if (
        residual.get("enabled", False) and residual.get("balanced_profile", False)
    ) else ("risk", "coverage")
    if primary not in allowed:
        raise ValueError("primary_profile is not enabled: " + str(primary))
    root = (resolve(run_dir) if run_dir else PROJECT_ROOT / "runs" /
            cfg["data"]["name"] / cfg["model"]["arch"] / cfg["exp"] /
            ("trial_" + str(trial)))
    if extension.get("require_archived_training_config", False):
        archived = resolve(extension["base_config"]).resolve()
        if archived.parent != root.resolve():
            raise ValueError("base_config must be the archived YAML inside --run-dir; do not mix old checkpoints and new splits")
        cfg["visual_support"]["archived_training_config_sha256"] = sha256(archived)
    checkpoint = root / "ckpt" / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))
    return cfg, root, checkpoint


def extraction_signature(cfg):
    # Includes source hash of the model implementation, not split paths.
    # Prevent memory/query encoding from silently using changed source files.
    model_sources = {}
    for name in ("models/maple.py", "models/maple_clip.py", "models/maple_model.py",
                 "models/__init__.py", "models/simple_tokenizer.py",
                 "models/bpe_simple_vocab_16e6.txt.gz", "loader/transforms.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            model_sources[name] = sha256(path)
    content = {"model": cfg["model"], "transform": cfg["data"].get("transform", "clip"),
               "hierarchy_sha256": sha256(resolve(cfg["data"]["hierarchy"])),
               "model_sources": model_sources}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


def load_model(cfg, checkpoint):
    import torch
    from loader.hierdata import _load_hierarchy
    from models import get_model
    from .core import validate_meta

    if not torch.cuda.is_available():
        raise RuntimeError("This MaPLe fp16 extraction requires your CUDA ProTeCt environment")
    # _load_hierarchy reads tree.npy only; it builds NO dataset loader.
    hierarchy = _load_hierarchy(cfg["data"])
    names = hierarchy["param_names"]
    parents = [names[int(i)] for i in hierarchy["intnl_nodes"][0]]
    leaves = [names[int(i)] for i in hierarchy["leaf_nodes"]]
    mapping = hierarchy["sublabels"][:, 0].long().tolist()
    meta = {"parent_names": parents, "leaf_names": leaves, "leaf_to_parent": mapping}
    validate_meta(meta)
    if parents != EXPECTED_PARENTS:
        raise ValueError("Unexpected parent ID order: {}".format(parents))
    if len(leaves) != int(cfg["data"].get("num_known_leaves", 17)):
        raise ValueError("Tree leaf count differs from configuration")
    device = torch.device("cuda")
    model = get_model(cfg["model"], leaves).to(device)
    # Only load the user's own trusted state-dict checkpoint.
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    core = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        texts = {
            "parent": core.encode_text(parents, normalize=True).float(),
            "leaf": core.encode_text(leaves, normalize=True).float(),
        }
        scale = float(core.model.logit_scale.exp().float().item())
    return core, texts, meta, device, scale


def _manifest(cfg, split, meta):
    if split not in SPLITS:
        raise ValueError("Unsupported split {}".format(split))
    kind = SPLITS[split]
    data = cfg["data"]
    if kind == "intra":
        root = resolve(data["full_data_root"])
    elif kind == "extra":
        root = resolve(data["ood_root"])
    else:
        root = resolve(data["data_root"])
    normal = lambda x: "".join(c for c in x.lower() if c.isalnum())
    known_species = {normal(name) for name in meta["leaf_names"]}
    entries = []
    with resolve(data[split]).open("r", encoding="utf-8-sig") as stream:
        for line_num, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.strip().rsplit(",", 2)
            if len(fields) != 3:
                raise ValueError("{}:{} requires path,label,index".format(split, line_num))
            relative, label, _ = fields
            label = int(label)
            full = Path(relative) if Path(relative).is_absolute() else root / relative
            if not full.is_file():
                raise FileNotFoundError("{}:{} missing {}".format(split, line_num, full))
            if kind == "known":
                if not 0 <= label < len(meta["leaf_names"]):
                    raise ValueError("{} has invalid known leaf ID".format(split))
                true_l, true_p = label, meta["leaf_to_parent"][label]
                if normal(full.parent.name) != normal(meta["leaf_names"][label]):
                    raise ValueError("{} path/leaf ID mismatch: {} -> {}".format(
                        split, relative, meta["leaf_names"][label]))
                if normal(full.parent.parent.name) != normal(meta["parent_names"][true_p]):
                    raise ValueError("{} known path/parent ID mismatch".format(split))
            elif kind == "intra":
                if not 0 <= label < len(meta["parent_names"]):
                    raise ValueError("{} has invalid parent ID".format(split))
                true_l, true_p = None, label
                if full.parent.parent.name != meta["parent_names"][label]:
                    raise ValueError("{} path/parent ID mismatch".format(split))
                if normal(full.parent.name) in known_species:
                    raise ValueError("{} contains a known-tree species marked intra unknown".format(split))
            else:
                if label != -1:
                    raise ValueError("{} extra label must be -1".format(split))
                true_l, true_p = None, None
            entries.append({"status": kind, "split": split, "dataset_index": len(entries),
                            "path": relative, "resolved_path": str(full.resolve()),
                            "source": full.parent.name, "true_leaf": true_l,
                            "true_parent": true_p, "image_sha256": sha256(full)})
    if not entries:
        raise ValueError("Empty split: {}".format(split))
    return entries


def extract(cfg, split, model, texts, meta, device):
    """Full deterministic split pass: no augmentation, no episode sampler.

    Each image is encoded ONCE; parent, leaf and visual evidence use that same
    feature. Input content hashes allow train/validation/test overlap checks.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from loader.transforms import get_transform
    from loader.utils import default_loader

    entries = _manifest(cfg, split, meta)
    transform = get_transform(cfg["data"].get("transform", "clip"), False)

    class Images(Dataset):
        def __len__(self):
            return len(entries)

        def __getitem__(self, index):
            return transform(default_loader(entries[index]["resolved_path"])), index

    loader = DataLoader(Images(), batch_size=int(cfg["data"].get("eval_batch_size", 16)),
                        num_workers=int(cfg["data"].get("n_workers", 4)),
                        shuffle=False, drop_last=False, pin_memory=True)
    features, pc, lc, seen = [], [], [], []
    with torch.no_grad():
        for step, (images, indices) in enumerate(loader):
            f = model.encode_image(images.to(device), normalize=True).float()
            features.append(f.cpu().numpy())
            pc.append((f @ texts["parent"].t()).cpu().numpy())
            lc.append((f @ texts["leaf"].t()).cpu().numpy())
            seen.extend(indices.tolist())
            if step == 0 or (step + 1) % 25 == 0:
                print("[{}] {}/{} images".format(split, len(seen), len(entries)), flush=True)
    if seen != list(range(len(entries))):
        raise RuntimeError("Feature order does not match the split manifest")
    return entries, np.concatenate(features), np.concatenate(pc), np.concatenate(lc)


def assert_disjoint(rows, previous_hashes, description):
    hashes = [r["image_sha256"] for r in rows]
    overlap = set(hashes).intersection(previous_hashes)
    if overlap:
        raise ValueError("{}: {} identical image(s) cross split/artifact boundaries; remove leakage".format(description, len(overlap)))
    if len(set(hashes)) != len(hashes):
        raise ValueError("{}: duplicate image bytes within evaluation inputs; deduplicate before calibration/testing".format(description))


def load_bank(folder, cfg, checkpoint):
    folder = Path(folder)
    info = read_json(folder / "memory.json")
    path = folder / "memory.npz"
    if info["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Memory/checkpoint mismatch; build a new memory")
    if info["extraction_signature"] != extraction_signature(cfg):
        raise RuntimeError("Model/transform/tree changed after memory extraction")
    if info["memory_sha256"] != sha256(path):
        raise RuntimeError("Memory file hash mismatch")
    with np.load(str(path), allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    return bank, info
