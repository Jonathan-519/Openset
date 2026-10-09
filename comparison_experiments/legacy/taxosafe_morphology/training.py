"""Separate R01/L01 optimization; image encoder and D05 remain frozen."""
import copy
import hashlib
import math
import time

import torch
from torch.nn import functional as F

from taxosafe_support.pipeline import seed_all
from .support_bank import candidate_indices, make_bank
from .verifier import SpatialVerifier


def fit_spatial(view, spatial, episodes, meta, cfg, level, device="cpu"):
    if level not in ("parent", "leaf") or episodes["report"].get("fit_split") != "known_train":
        raise ValueError("Spatial fitting accepts only a declared level and known TRAIN episodes")
    if any(row["split"] != "train" or row["status"] != "known" for row in view["records"]):
        raise ValueError("Unknown/DEV/TEST rows cannot enter spatial gradients")
    if spatial["image_sha256"] != view["image_sha256"]:
        raise ValueError("Spatial training rows differ from D05 identities")
    seed_all(cfg["seed"])
    settings = cfg["training"]
    tokens, positions = spatial["tokens"], spatial["positions"].to(device)
    labels, hashes = view["labels"], view["image_sha256"]
    model = SpatialVerifier(tokens.shape[-1], len(meta["leaf_names"]), settings["adapter_dim"], settings["hidden"]).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    examples = episodes["examples"][level]
    weight = torch.tensor([row["weight"] for row in examples], dtype=torch.double)
    generator = torch.Generator().manual_seed(cfg["seed"])
    order = torch.multinomial(weight, settings["steps"] * settings["batch_size"], replacement=True, generator=generator)
    histories, sums, gradients = [], [0., 0., 0.], dict(adapter=0., delta_head=0., classifier=0.)
    max_initial_delta, max_capacity_error = 0., 0.
    start = time.perf_counter()
    for step in range(settings["steps"]):
        optimizer.zero_grad()
        batch = order[step * settings["batch_size"]:(step + 1) * settings["batch_size"]]
        for index in batch.tolist():
            row = examples[index]
            episode = episodes["report"]["episodes"][row["episode"]]
            refs = torch.tensor(episode["reference_indices"], dtype=torch.long)
            refs = refs[candidate_indices(labels[refs], meta, level, row["candidate"])]
            if not len(refs) or hashes[row["query"]] in {hashes[i] for i in refs.tolist()}:
                raise ValueError("Empty candidate support or leaked query identity")
            delta, pooled, detail = model.evidence(tokens[row["query"]].to(device), tokens[refs].to(device),
                                                   labels[refs].to(device), positions, cfg["matching"])
            if step == 0:
                max_initial_delta = max(max_initial_delta, abs(float(delta.detach())))
            max_capacity_error = max(max_capacity_error, float(detail["capacity_error"].detach().max()))
            score = delta + delta.new_tensor(row["base"])
            bce = F.binary_cross_entropy_with_logits(score, score.new_tensor(row["target"]))
            ce = delta.new_zeros(())
            if row["ce_allowed"] and settings["auxiliary_ce"] > 0:
                active = torch.tensor(episode["active_leaves"], device=device, dtype=torch.long)
                truth = int(labels[row["query"]])
                target = (active == truth).nonzero(as_tuple=True)[0]
                if len(target) != 1:
                    raise ValueError("Local CE cannot restore a withheld class")
                logits = model.local_classifier(pooled)[active]
                ce = F.cross_entropy(logits[None], target)
            loss = bce + settings["auxiliary_ce"] * ce
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite spatial training loss")
            (loss / len(batch)).backward()
            sums[0] += float(loss.detach()); sums[1] += float(bce.detach()); sums[2] += float(ce.detach())
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise ValueError("Nonfinite spatial gradient")
                kind = "delta_head" if name.startswith("delta.") else "classifier" if name.startswith("local_classifier.") else "adapter"
                gradients[kind] += float(parameter.grad.detach().norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 0 and max_initial_delta != 0.:
            raise AssertionError("Zero-initialized correction did not reproduce the D05 control")
        if (step + 1) % settings["log_every"] == 0 or step + 1 == settings["steps"]:
            count = (step + 1) * settings["batch_size"]
            history = dict(step=step + 1, sampled_pairs=count, loss=sums[0]/count,
                           bce=sums[1]/count, auxiliary_ce=sums[2]/count,
                           averaging="cumulative_sampled_pairs",
                           seconds=time.perf_counter()-start)
            histories.append(history)
            print("{} spatial step {}/{} BCE {:.6f}, CE {:.6f}, elapsed {:.1f}s".format(
                level, step + 1, settings["steps"], history["bce"], history["auxiliary_ce"], history["seconds"]), flush=True)
    changes = {k: float((v.detach().cpu()-initial[k]).double().norm()) for k,v in model.state_dict().items()}
    state = model.export_state()
    bank = make_bank(tokens, spatial["positions"], view["raw_clip"], labels, hashes, meta, cfg["support"]["references_per_leaf"])
    report = dict(training_execution="known_TRAIN_spatial_gradient_fit", level=level,
                  optimizer_steps=settings["steps"], gradient_splits=["train"],
                  initialized_from="zero_correction_over_original_D05_with_frozen_reference_spatial_tokens",
                  new_gradient_updates=True, frozen_encoder_updated=False, frozen_d05_updated=False,
                  new_spatial_adapter_updated=True, checkpoint_selection="final_predeclared_step_no_DEV_selection",
                  local_auxiliary_ce_uses_withheld_classes=False, initial_control_max_abs_delta=max_initial_delta,
                  sampled_pair_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                  parameter_delta_l2=math.sqrt(sum(v*v for v in changes.values())), changed_tensors=changes,
                  gradient_l2_sum=gradients, max_capacity_error=max_capacity_error,
                  training_settings=copy.deepcopy(settings), matching_settings=copy.deepcopy(cfg["matching"]),
                  fit_image_sha256=list(hashes), episode_report=copy.deepcopy(episodes["report"]),
                  history=histories, seconds=time.perf_counter()-start,
                  test_used_for_fitting=False, unknown_images_used_for_gradients=False)
    if not report["parameter_delta_l2"] > 0:
        raise ValueError("Spatial optimizer made no parameter change")
    return dict(verifier=state, support_bank=bank, level=level, matching=copy.deepcopy(cfg["matching"])), report


@torch.no_grad()
def score_spatial(model_state, spatial, candidates, meta, device="cpu"):
    """Truth-free inference from serialized states, without any statistical fit."""
    from .matching import DESCRIPTORS
    model = SpatialVerifier.restore(model_state["verifier"], device)
    bank, level = model_state["support_bank"], model_state["level"]
    if bank["fit_split"] != "known_train" or bank["meta"] != meta:
        raise ValueError("Invalid spatial inference support provenance")
    if set(bank["image_sha256"]) & set(spatial["image_sha256"]):
        raise ValueError("Inference query overlaps TRAIN support")
    if not torch.equal(bank["positions"], spatial["positions"]):
        raise ValueError("Inference token grid differs from trained reference grid")
    positions, refs = bank["positions"].to(device), bank["tokens"].to(device)
    labels = bank["labels"].to(device)
    corrections, records = [], []
    for i, candidate in enumerate(candidates):
        selected = candidate_indices(labels, meta, level, int(candidate))
        if not len(selected):
            raise ValueError("Candidate has no real TRAIN spatial reference")
        delta, _, evidence = model.evidence(spatial["tokens"][i].to(device), refs[selected], labels[selected],
                                            positions, model_state["matching"])
        corrections.append(float(delta))
        best = evidence["selected_reference"]
        bank_index = int(selected[best])
        hashes = [bank["image_sha256"][int(index)] for index in selected]
        records.append(dict(level=level, candidate=int(candidate), delta=float(delta),
            selected_reference_sha256=bank["image_sha256"][bank_index],
            selected_mode_leaf=evidence["selected_mode"],
            aggregate_reference_sha256=[hashes[j] for j in evidence["aggregate_references"]],
            real_reference_sha256=hashes,
            per_reference=[dict(reference_sha256=hashes[j], **{name:float(value) for name,value in zip(DESCRIPTORS, desc)})
                           for j, desc in enumerate(evidence["descriptors"])],
            query_patch_cost=evidence["query_patch_cost"][best].cpu().tolist(),
            reference_patch_cost=evidence["reference_patch_cost"][best].cpu().tolist(),
            target_patch=evidence["target_patch"][best].cpu().tolist(),
            max_capacity_error=float(evidence["capacity_error"].max()),
            matched_mass=float(evidence["matched_mass"][best]), discarded_patch_mass=0.,
            interpretation="cost/coverage evidence only; no anatomical localization or visibility claim"))
    return torch.tensor(corrections, dtype=torch.float64), records
