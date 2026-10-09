"""Known-TRAIN-only feature fine tuning and hierarchical novelty episodes."""
import copy
from collections import Counter
import time
import numpy as np
import torch
from torch.nn import functional as F
from taxosafe_support import calibration as base
from taxosafe_discovery.features import tensor_hash
from .memory import build_memory,to_device
from .model import JointModel
from .protocol import object_hash


def labels_for(group):
    rows={x["image_sha256"]:x for x in base.unique_records(group["records"])}
    if any(rows[h]["split"]!="train" or rows[h]["status"]!="known" for h in group["image_sha256"]):
        raise ValueError("Gradient fitting accepts only original known TRAIN images")
    return torch.tensor([rows[h]["true_leaf"] for h in group["image_sha256"]],dtype=torch.long)


def episode_masks(labels,meta,rng):
    """Class/parent removal has matched positive controls and no count feature."""
    mapping=np.asarray(meta["leaf_to_parent"])
    c,p=len(mapping),len(meta["parent_names"])
    mask=np.ones((len(labels),c),dtype=bool);target=[];kinds=[]
    for i,leaf in enumerate(labels.tolist()):
        parent=int(mapping[leaf]);siblings=np.flatnonzero((mapping==parent)&(np.arange(c)!=leaf))
        options=["known","extra"]+(["near"] if len(siblings) else [])
        kind=options[int(rng.integers(len(options)))]
        if kind=="near":
            mask[i,leaf]=False;terminal=c+parent
        elif kind=="extra":
            mask[i,mapping==parent]=False;terminal=c+p
        else:
            if len(siblings) and rng.random()<.5:
                mask[i,int(rng.choice(siblings))]=False
            else:
                others=np.flatnonzero(np.arange(p)!=parent)
                mask[i,mapping==int(rng.choice(others))]=False
            terminal=leaf
        target.append(terminal);kinds.append(kind)
    return torch.from_numpy(mask),torch.tensor(target,dtype=torch.long),kinds


def _balanced_indices(labels,fold_ids,fold,size,rng):
    available={int(c):( (labels==c)&(fold_ids==fold)).nonzero(as_tuple=False).flatten().tolist() for c in labels.unique()}
    available={c:v for c,v in available.items() if v}
    if not available:
        return None
    parents=list(available)
    return torch.tensor([int(rng.choice(available[int(rng.choice(parents))])) for _ in range(size)],dtype=torch.long)


def _tail_distribution(fine,parent,labels):
    joined=torch.cat((fine,parent),1)
    classes=labels.unique()
    means=torch.stack([joined[labels==c].mean(0) for c in classes])
    centered=torch.cat([joined[labels==c]-joined[labels==c].mean(0) for c in classes])
    variance=centered.square().mean(0).clamp(min=1e-5)
    return dict(means=means,variance=variance,dimension=fine.shape[1])


def _tail_samples(distribution,count,generator):
    """VOS-inspired diagonal Gaussian tail features, drawn only from TRAIN.

    This is a declared adaptation, not a reproduction of VOS. Synthetic vectors
    are not real unseen species and are never counted in evaluation metrics.
    """
    means,variance=distribution["means"],distribution["variance"]
    chosen=torch.randint(len(means),(count,),generator=generator)
    noise=torch.randn(count,8,means.shape[1],generator=generator)
    # Low Gaussian likelihood: select the largest standardized residual in each pool.
    pick=noise.square().sum(-1).argmax(1)
    raw=means[chosen]+2.0*noise[torch.arange(count),pick]*variance.sqrt()
    d=distribution["dimension"]
    return F.normalize(raw[:,:d],dim=-1),F.normalize(raw[:,d:],dim=-1)


def fit(cache,meta,arm,cfg,device="cpu"):
    started=time.perf_counter()
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg["seed"])
    rng=np.random.default_rng(cfg["seed"])
    gen=torch.Generator().manual_seed(cfg["seed"]+178)
    group=cache["groups"]["train"]
    y=labels_for(group);hashes=group["image_sha256"]
    if len(meta["parent_names"])<2:
        raise ValueError("Hierarchical parent removal needs at least two parents")
    memory=build_memory(group["fine"],group["parent"],y,hashes,meta,cfg["support"],cfg["seed"])
    virtual_distribution=_tail_distribution(group["fine"],group["parent"],y) if arm["virtual"] else None
    settings=cfg["training"]
    model=JointModel(group["fine"].shape[1],meta,arm,settings).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    fine,parent=group["fine"].to(device),group["parent"].to(device)
    text=cache["text"].to(device);banks=[to_device(b,device) for b in memory["banks"]]
    mapping=torch.tensor(meta["leaf_to_parent"],dtype=torch.long,device=device)
    teacher=F.normalize(fine,dim=-1)@F.normalize(text[:model.c],dim=-1).T*cache["logit_scale"]
    history=[];episodes=Counter();sampled=[];synthetic=0
    model.train()
    for step in range(settings["steps"]):
        fold=step%cfg["support"]["folds"]
        index=_balanced_indices(y,memory["fold_ids"],fold,settings["batch_size"],rng)
        if index is None:
            available=sorted(set(memory["fold_ids"].tolist()))
            fold=available[step%len(available)]
            index=_balanced_indices(y,memory["fold_ids"],fold,settings["batch_size"],rng)
        current_y=y[index];active,target,kinds=episode_masks(current_y,meta,rng)
        episodes.update(kinds);sampled.append(object_hash([fold,index.tolist(),kinds]))
        index_device=index.to(device);target=target.to(device);active=active.to(device)
        qf,qp=fine[index_device],parent[index_device]
        episode_logits,_=model(qf,qp,banks[fold+1],text,active)
        # Balance terminal kinds and, within each kind, give parents/leaves equal influence.
        losses=F.cross_entropy(episode_logits,target,reduction="none")
        weights=torch.zeros_like(losses)
        target_groups=current_y.tolist()
        for kind in sorted(set(kinds)):
            groups=sorted(set((target_groups[i] if kind=="known" else meta["leaf_to_parent"][target_groups[i]]) for i,k in enumerate(kinds) if k==kind))
            for node in groups:
                ids=[i for i,k in enumerate(kinds) if k==kind and (target_groups[i] if kind=="known" else meta["leaf_to_parent"][target_groups[i]])==node]
                weights[ids]=1.0/(len(set(kinds))*len(groups)*len(ids))
        episodic=(losses*weights).sum()
        full,z=model(qf,qp,banks[fold+1],text)
        target_known=current_y.to(device)
        known_ce=F.cross_entropy(full,target_known)
        class_logits=z@F.normalize(text[:model.c],dim=-1).T*cache["logit_scale"]
        closed_ce=F.cross_entropy(class_logits,target_known)
        loss=episodic+known_ce+settings["classification"]*closed_ce
        distill=torch.zeros((),device=device);anchor=torch.zeros((),device=device)
        if arm["distill"]:
            distill=F.kl_div(F.log_softmax(class_logits/2.,dim=-1),F.softmax(teacher[index_device]/2.,dim=-1),reduction="batchmean")*4.
            anchor=(z-F.normalize(qf,dim=-1)).square().sum(1).mean()
            loss=loss+settings["distillation"]*distill+settings["feature_anchor"]*anchor
        correct_score=class_logits.gather(1,target_known[:,None]).squeeze(1)
        siblings=(mapping[None]==mapping[target_known,None])&(torch.arange(model.c,device=device)[None]!=target_known[:,None])
        sibling_exists=siblings.any(1)
        if bool(sibling_exists.any()):
            rival=class_logits.masked_fill(~siblings,-10000.).max(1).values
            loss=loss+settings["sibling_margin"]*F.relu(1.+rival[sibling_exists]-correct_score[sibling_exists]).mean()
        adv=torch.zeros((),device=device);virtual=torch.zeros((),device=device)
        if arm["adversarial"]:
            adv=F.cross_entropy(model.adversarial_logits(qf),mapping[target_known])
            loss=loss+settings["adversarial"]*adv
        if arm["virtual"]:
            vf,vp=_tail_samples(virtual_distribution,max(4,settings["batch_size"]//4),gen)
            out,_=model(vf.to(device),vp.to(device),banks[0],text)
            virtual=F.cross_entropy(out,torch.full((len(vf),),model.c+model.p,dtype=torch.long,device=device))
            loss=loss+settings["virtual"]*virtual;synthetic+=len(vf)
        if not bool(torch.isfinite(loss)):raise ValueError("Non-finite training loss")
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
        if step==0 or (step+1)%settings["log_every"]==0 or step+1==settings["steps"]:
            item=dict(step=step+1,loss=float(loss.detach()),episode=float(episodic.detach()),known=float(known_ce.detach()),
                      closed_ce=float(closed_ce.detach()),distill=float(distill.detach()),anchor=float(anchor.detach()),
                      adversarial=float(adv.detach()),virtual=float(virtual.detach()))
            history.append(item);print(arm["id"],item,flush=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    delta={k:float((v.float()-initial[k].float()).square().sum().sqrt()) for k,v in state.items() if v.is_floating_point()}
    adapter_delta=sum(v*v for k,v in delta.items() if k.startswith("adapter."))**.5
    report=dict(training_execution="known_TRAIN_joint_state_gradient_fit",optimizer_steps=settings["steps"],
        feature_initialization="exact C00 parent/fine embeddings; zero residual fine adapter",frozen_encoder_updated=False,
        fine_feature_adapter_updated=arm["adapter"],coarse_feature_adapter_updated=False,old_C00_checkpoint_modified=False,
        gradient_splits=["train"],unknown_images_used_for_gradients=False,test_used_for_fitting=False,
        checkpoint_selection="final_predeclared_step",parameter_delta_l2=sum(v*v for v in delta.values())**.5,
        fine_adapter_delta_l2=adapter_delta,
        joint_head_delta_l2=sum(v*v for k,v in delta.items() if k.startswith(("leaf_head.","parent_head.","root_head.")))**.5,
        tensor_delta_l2=delta,history=history,fit_image_sha256=list(hashes),episode_counts=dict(episodes),
        sampled_episode_sha256=object_hash(sampled),held_query_images_excluded_from_support=True,
        withheld_leaf_and_parent_removed_from_support=True,matched_positive_removal_controls=True,
        support_count_feature_used=False,encoder_unseen_class_claim=False,
        synthetic_feature_count=synthetic,synthetic_features_count_as_real_unknowns=False,
        seconds=time.perf_counter()-started)
    if report["parameter_delta_l2"]<=0:raise ValueError("No actual parameter update")
    if arm["adapter"] and adapter_delta<=0:raise ValueError("Fine adapter did not actually update")
    model.eval().cpu()
    payload=dict(arguments=model.arguments,state=state,memory=memory,text=cache["text"],meta=meta,
                 train_feature_sha256=tensor_hash(group["fine"]),report=report)
    return payload,report


@torch.no_grad()
def score(cache,payload,device="cpu"):
    model=JointModel(**payload["arguments"]).to(device).eval()
    model.load_state_dict(payload["state"],strict=True)
    bank=to_device(payload["memory"]["banks"][0],device);text=payload["text"].to(device)
    groups={}
    for split,group in cache["groups"].items():
        values=[]
        for start in range(0,len(group["fine"]),128):
            logits,_=model(group["fine"][start:start+128].to(device),group["parent"][start:start+128].to(device),bank,text)
            if not bool(torch.isfinite(logits).all()):raise ValueError("Non-finite joint scores")
            values.extend(logits.cpu().tolist())
        by_hash=dict(zip(group["image_sha256"],values))
        groups[split]=[dict(copy.deepcopy(row),joint_logits=by_hash[row["image_sha256"]],
            joint_score_order="leaves,parent_unknown,root",log_probs_origin="immutable_C00_diagnostic") for row in group["records"]]
    return groups
