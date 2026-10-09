"""Direct C00 feature extraction using the checkpoint's original eval batch."""
import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from taxosafe_support import pipeline as support
from taxosafe_support import protocol as base_protocol
from taxosafe_support import calibration as base
from taxosafe_support.membership_calibration import RAW_FIELDS
from taxosafe_refine.importer import load_reference
from taxosafe_refine.pipeline import load_training_rows, _check_baseline_development
from taxosafe_discovery.features import tensor_hash


def source_rows(info, stage):
    if stage == "train":
        rows, audit = load_training_rows(SimpleNamespace(**info))
        return {"train": rows}, {"train": audit}
    if stage not in ("development", "test"):
        raise ValueError("Unknown feature stage")
    forbidden = set(info["training"]["audit"]["train"]["image_hashes"])
    names = set()
    if stage == "test":
        for split, audit in info["calibration"]["audit"].items():
            forbidden.update(audit["image_hashes"])
            if split in ("val_intra", "val_extra"):
                names.update(audit["sources"])
    groups, audit = support.load_stage_rows(info["config"], "calibrate" if stage=="development" else "test",
        info["meta"], forbidden_hashes=forbidden, forbidden_sources=names)
    if any(name in info["audit"] and item != info["audit"][name] for name,item in audit.items()):
        raise ValueError("Input images or manifests differ from immutable C00 audit")
    return groups,audit


def audit_images(groups):
    problems=[]
    for split,rows in groups.items():
        for row in base.unique_records(rows):
            path=Path(row["resolved_path"])
            try:
                if not path.is_file() or base_protocol.file_hash(path)!=row["image_sha256"]:
                    raise ValueError("Missing/changed image bytes")
                with Image.open(path) as image:
                    image.verify()
            except (OSError,ValueError) as error:
                problems.append(dict(split=split,path=str(path),error=str(error)))
    return dict(valid=not problems,problems=problems,
                image_count=sum(len(base.unique_records(v)) for v in groups.values()))


@torch.no_grad()
def collect(info, groups, device):
    source=load_reference(info["directory"],torch.device(device))
    if source.binding!=info["binding"]:
        raise ValueError("C00 binding changed before extraction")
    text=source.encoder.text_features()
    output={}
    for split,raw in groups.items():
        rows=base.unique_records(raw)
        fine,parent,scored,seen=[],[],[],[]
        for images,_,indices in support.make_loader(rows,source.config,source.meta,training=False):
            encoded=source.encoder.encode(images.to(device),text_features=text)
            evidence=source.evidence(encoded,source.bank)
            batch=base.raw_records([rows[i] for i in indices.tolist()],
                {"log_probs":evidence["log_probs"].cpu().numpy()},encoded["leaf_logits"].cpu().numpy(),source.meta)
            diagnostic={key:evidence[key].cpu().tolist() for key in RAW_FIELDS}
            for index,row in enumerate(batch):
                row["support_evidence"]={key:value[index] for key,value in diagnostic.items()}
            scored.extend(batch);seen.extend(indices.tolist())
            fine.append(encoded["fine"].float().cpu());parent.append(encoded["parent"].float().cpu())
        if seen!=list(range(len(rows))):
            raise ValueError("Reference image order changed")
        by_hash={x["image_sha256"]:x for x in scored}
        output[split]=dict(records=[dict(by_hash[x["image_sha256"]],**x) for x in raw],
            image_sha256=[x["image_sha256"] for x in rows],fine=torch.cat(fine),parent=torch.cat(parent))
        print("C00 cache {}: {} images; original eval batch {}".format(split,len(rows),source.config["data"].get("eval_batch_size",16)),flush=True)
    result=dict(groups=output,text=text.detach().float().cpu(),meta=copy.deepcopy(source.meta),
                source_binding=copy.deepcopy(source.binding),reference_eval_batch_size=source.config["data"].get("eval_batch_size",16),
                logit_scale=float(source.encoder.backbone.model.logit_scale.exp().float()),
                frozen_encoder_updated=False,feature_origin="C00 trained parent/fine TokenBranch outputs")
    if all(k.startswith("val_") for k in groups):
        # The legacy reproduction helper deduplicates through the refine schema.
        # It requires this unused field; adapt a copy without changing C00 scores.
        result["source_reproduction"]=_check_baseline_development(source,
            {k:[dict(row,reconstruction_score=0.) for row in v["records"]] for k,v in output.items()})
    source.encoder.cpu();source.evidence.cpu();source.bank.to("cpu")
    return result


def summary(cache):
    return dict(feature_origin=cache["feature_origin"],reference_eval_batch_size=cache["reference_eval_batch_size"],
        frozen_encoder_updated=False,source_reproduction=cache.get("source_reproduction"),
        groups={name:dict(count=len(value["image_sha256"]),fine_shape=list(value["fine"].shape),
            fine_sha256=tensor_hash(value["fine"]),parent_sha256=tensor_hash(value["parent"])) for name,value in cache["groups"].items()})
