"""Read global and spatial MaPLe tokens in one frozen image forward pass."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from taxosafe_visual.runtime import _manifest, assert_disjoint


def spatial_tokens(output, visual, pool_side=4):
    """MaPLe appends prompt tokens AFTER patches; exclude CLS and all prompts."""
    sequence = output[0] if isinstance(output, (list, tuple)) else output
    count = int(visual.positional_embedding.shape[0]) - 1
    side = int(math.isqrt(count))
    if side * side != count or sequence.ndim != 3 or len(sequence) < count + 1:
        raise ValueError("Unexpected MaPLe visual token layout")
    patches = sequence[1:1 + count].permute(1, 0, 2)
    patches = visual.ln_post(patches)
    if visual.proj is not None:
        patches = patches @ visual.proj
    b, _, d = patches.shape
    patches = patches.float().reshape(b, side, side, d).permute(0, 3, 1, 2)
    patches = F.adaptive_avg_pool2d(patches, (int(pool_side), int(pool_side)))
    return F.normalize(patches.flatten(2).transpose(1, 2), dim=-1)


def extract_spatial(cfg, splits, model, texts, meta, device, forbidden=(), pool_side=4):
    from torch.utils.data import Dataset, DataLoader
    from loader.transforms import get_transform
    from loader.utils import default_loader
    visual = model.model.image_encoder
    if visual.__class__.__name__ != "VisionTransformer_MaPLe":
        raise ValueError("Spatial extractor currently supports ViT MaPLe only")
    rows, gs, ps, pcs, lcs = [], [], [], [], []
    seen = set(forbidden)
    captured = []
    def hook(module, inputs, output):
        captured.append(spatial_tokens(output, visual, pool_side))
    handle = visual.transformer.register_forward_hook(hook)
    try:
        for split in splits:
            entries = _manifest(cfg, split, meta)
            assert_disjoint(entries, seen, split)
            seen.update(r["image_sha256"] for r in entries)
            transform = get_transform(cfg["data"].get("transform", "clip"), False)
            class Images(Dataset):
                def __len__(self):
                    return len(entries)
                def __getitem__(self, index):
                    return transform(default_loader(entries[index]["resolved_path"])), index
            loader = DataLoader(Images(), batch_size=int(cfg["data"].get("eval_batch_size", 16)),
                                num_workers=int(cfg["data"].get("n_workers", 4)), shuffle=False)
            order = []
            with torch.no_grad():
                for images, indices in loader:
                    captured.clear()
                    global_features = model.encode_image(images.to(device), normalize=True).float()
                    if len(captured) != 1:
                        raise RuntimeError("Expected exactly one visual transformer forward")
                    gs.append(global_features.cpu().numpy())
                    ps.append(captured[0].cpu().numpy())
                    pcs.append((global_features @ texts["parent"].T).cpu().numpy())
                    lcs.append((global_features @ texts["leaf"].T).cpu().numpy())
                    order.extend(indices.tolist())
            if order != list(range(len(entries))):
                raise RuntimeError("Feature and manifest order differ")
            rows.extend(entries)
            print("Cached {}: {} images".format(split, len(entries)), flush=True)
    finally:
        handle.remove()
    data = {"global": np.concatenate(gs), "patches": np.concatenate(ps).astype(np.float16),
            "parent_cosine": np.concatenate(pcs), "leaf_cosine": np.concatenate(lcs),
            "parent_text": texts["parent"].cpu().numpy(), "leaf_text": texts["leaf"].cpu().numpy()}
    return rows, data
