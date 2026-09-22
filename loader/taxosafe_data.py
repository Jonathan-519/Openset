"""Data-loader composition for all TaxoSafe train/calibration/test splits."""

import copy
import logging
import os

import torch.utils.data as data

from .collate import get_collate_fn
from .hierdata import HierDataLoader
from .img_flist import ImageFilelist
from .transforms import get_transform


logger = logging.getLogger("mylogger")

TAXOSAFE_SPLITS = {
    "train",
    "train_reference",
    "val_known",
    "test_known",
    "oe_train",
    "val_intra",
    "test_intra",
    "val_extra",
    "test_extra",
    "dev_unknown",
    "test_unknown",
}


def _plain_loader(
    root_dir,
    flist,
    transform_name,
    augment,
    batch_size,
    num_workers,
    shuffle,
):
    if not root_dir or not os.path.isdir(root_dir):
        raise FileNotFoundError(
            "Image root does not exist: {}".format(root_dir)
        )
    if not flist or not os.path.isfile(flist):
        raise FileNotFoundError(
            "Data list does not exist: {}".format(flist)
        )

    dataset = ImageFilelist(
        root_dir=root_dir,
        flist=flist,
        transform=get_transform(transform_name, augment),
    )
    return data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        sampler=None,
        drop_last=False,
        collate_fn=get_collate_fn(False),
        pin_memory=True,
        num_workers=int(num_workers),
    )


def TaxoSafeDataLoader(cfg, splits, batch_size):
    """Build loaders while preserving the eight public TaxoSafe split names.

    ``HierDataLoader`` internally expects ``train/val/test``. This adapter maps
    ``val_known`` and ``test_known`` to those names only while constructing the
    known loaders, then restores the explicit names. As a result, training code
    cannot accidentally confuse validation and test data.

    Legacy requests containing ``val`` or ``test`` are accepted and returned
    as ``val_known`` and ``test_known`` for compatibility.
    """
    requested = list(splits)
    requested_set = set(requested)
    legacy = requested_set.intersection({"val", "test"})
    unsupported = requested_set.difference(TAXOSAFE_SPLITS | {"val", "test"})
    if unsupported:
        raise ValueError(
            "Unsupported TaxoSafe splits: {}".format(sorted(unsupported))
        )

    known_cfg = copy.deepcopy(cfg)
    known_cfg["train"] = cfg.get("train")
    known_cfg["val"] = cfg.get("val_known", cfg.get("val"))
    known_cfg["test"] = cfg.get("test_known", cfg.get("test"))
    known_cfg.setdefault("seed", cfg.get("seed", 1))

    known_internal_splits = []
    if "train" in requested_set:
        known_internal_splits.append("train")
    if "val_known" in requested_set or "val" in legacy:
        known_internal_splits.append("val")
    if "test_known" in requested_set or "test" in legacy:
        known_internal_splits.append("test")

    if not known_internal_splits:
        raise ValueError(
            "At least one known split is required to load hierarchy metadata"
        )

    output = HierDataLoader(
        known_cfg,
        known_internal_splits,
        batch_size,
    )
    if "val" in output:
        output["val_known"] = output.pop("val")
    if "test" in output:
        output["test_known"] = output.pop("test")

    num_workers = int(cfg.get("n_workers", 4))
    eval_batch_size = int(cfg.get("eval_batch_size", 128))
    transform_name = cfg.get("transform", "imagenet")
    
    # ---------------------------------------------------------
    # Known taxonomy root
    # ---------------------------------------------------------
    data_root = cfg.get("data_root")
    
    # ---------------------------------------------------------
    # Legacy roots
    # 保留用于兼容以前的 TaxoSafe 配置
    # ---------------------------------------------------------
    full_root = cfg.get("full_data_root")
    ood_root = cfg.get("ood_root")
    
    # ---------------------------------------------------------
    # Split-specific physical roots
    # 新协议优先使用；旧配置自动 fallback
    # ---------------------------------------------------------
    near_dev_root = cfg.get(
        "near_dev_root",
        full_root,
    )
    
    near_test_root = cfg.get(
        "near_test_root",
        full_root,
    )
    
    ood_dev_root = cfg.get(
        "ood_dev_root",
        ood_root,
    )
    
    ood_test_root = cfg.get(
        "ood_test_root",
        ood_root,
    )
    
    oe_train_root = cfg.get(
        "oe_train_root",
        ood_root,
    )



    # Calibration-time reference bank.  Unlike ``train``, this loader is
    # deterministic, unaugmented and visits every training image once; it is
    # never used for gradient updates.
    if "train_reference" in requested_set:
        output["train_reference"] = _plain_loader(
            root_dir=data_root,
            flist=cfg.get("train"),
            transform_name=transform_name,
            augment=False,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    for split, root_dir in (
        ("val_intra", near_dev_root),
        ("test_intra", near_test_root),
    ):
        if split in requested_set:
            output[split] = _plain_loader(
                root_dir=root_dir,
                flist=cfg.get(split),
                transform_name=transform_name,
                augment=False,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

    for split, root_dir in (
        ("dev_unknown", near_dev_root),
        ("test_unknown", near_test_root),
    ):
        if split in requested_set:
            output[split] = _plain_loader(
                root_dir=root_dir,
                flist=cfg.get(split),
                transform_name=transform_name,
                augment=False,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

    if "oe_train" in requested_set:
        output["oe_train"] = _plain_loader(
            root_dir=oe_train_root,
            flist=cfg.get("oe_train"),
            transform_name=transform_name,
            augment=True,
            batch_size=int(batch_size),
            num_workers=num_workers,
            shuffle=True,
        )

    for split, root_dir in (
        ("val_extra", ood_dev_root),
        ("test_extra", ood_test_root),
    ):
        if split in requested_set:
            output[split] = _plain_loader(
                root_dir=root_dir,
                flist=cfg.get(split),
                transform_name=transform_name,
                augment=False,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

    logger.info("TaxoSafe loaders: %s", sorted(
        key for key in output if key in TAXOSAFE_SPLITS
    ))
    return output
