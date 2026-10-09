"""One learned decision over all leaves, parent-unknown states, and root.

Fine and coarse evidence remain separate. No truth, source filename, species
name, split, support count or episode type is an input to this network.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class JointModel(nn.Module):
    def __init__(self, dimension, meta, arm, settings):
        super().__init__()
        self.arguments = dict(dimension=dimension, meta=meta, arm=arm, settings=settings)
        self.c, self.p = len(meta["leaf_names"]), len(meta["parent_names"])
        self.register_buffer("mapping", torch.tensor(meta["leaf_to_parent"], dtype=torch.long))
        self.adapter_enabled = arm["adapter"]
        self.adapter = nn.Sequential(nn.Linear(dimension, settings["adapter_dim"]), nn.GELU(),
                                     nn.Linear(settings["adapter_dim"], dimension))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.adapter.requires_grad_(self.adapter_enabled)
        width = settings["hidden"]
        self.experts = nn.Embedding(self.p, 8) if arm["experts"] else None
        n = 16 if self.experts is not None else 8
        self.leaf_head = nn.Sequential(nn.Linear(n,width),nn.GELU(),nn.Linear(width,1))
        self.parent_head = nn.Sequential(nn.Linear(n,width),nn.GELU(),nn.Linear(width,1))
        self.root_head = nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,1))
        self.parent_adversary = nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,self.p))
        self.parent_adversary.requires_grad_(arm["adversarial"])

    def adapt(self, value):
        original = F.normalize(value.float(), dim=-1)
        return F.normalize(original + self.adapter(original),dim=-1) if self.adapter_enabled else original

    @staticmethod
    def top(values):
        # Missing second alternatives are represented by the first, never infinity.
        ordered = values.topk(min(2, values.shape[-1]), dim=-1).values
        first = ordered[...,0]
        second = ordered[...,-1].clamp(min=-1.0)
        return first, second

    def forward(self, fine, parent, bank, text, active=None):
        z, coarse = self.adapt(fine), F.normalize(parent.float(),dim=-1)
        bf = self.adapt(bank["fine"])
        bp = F.normalize(bank["parent"].float(),dim=-1)
        f = torch.einsum("bd,ckd->bck",z,bf).masked_fill(~bank["valid"][None],-10000.0)
        p = torch.einsum("bd,ckd->bck",coarse,bp).masked_fill(~bank["valid"][None],-10000.0)
        f1,f2 = self.top(f)
        p1,_ = self.top(p)
        if active is None:
            active = torch.ones(len(z),self.c,dtype=torch.bool,device=z.device)
        if active.shape != (len(z),self.c) or not bool(active.any(1).all()):
            raise ValueError("Every query must have nonempty active support")
        f1 = f1.masked_fill(~active,-10000.0)
        p1 = p1.masked_fill(~active,-10000.0)
        leaf_text = z @ F.normalize(text[:self.c].float(),dim=-1).T
        parent_text = coarse @ F.normalize(text[self.c:].float(),dim=-1).T
        parent_best, child_best, child_second, text_best, present = [],[],[],[],[]
        for j in range(self.p):
            children = self.mapping == j
            present.append(active[:,children].any(1))
            parent_best.append(p1[:,children].max(1).values)
            one,two = self.top(f1[:,children]);child_best.append(one);child_second.append(two)
            text_best.append(leaf_text[:,children].masked_fill(~active[:,children],-10000.0).max(1).values)
        parent_best, child_best, child_second, text_best, present = [torch.stack(v,1) for v in
            (parent_best,child_best,child_second,text_best,present)]
        pf = parent_best.clamp(min=-1.0)
        cf = child_best.clamp(min=-1.0)
        pgap = []
        for j in range(self.p):
            others = torch.arange(self.p,device=z.device) != j
            rival = parent_best[:,others].max(1).values.clamp(min=-1.0) if self.p>1 else torch.zeros(len(z),device=z.device)
            pgap.append(pf[:,j]-rival)
        pgap = torch.stack(pgap,1)
        lf = f1.clamp(min=-1.0)
        leaf_input = torch.stack((lf, f2, lf-f2, lf-cf[:,self.mapping], pf[:,self.mapping],
                                 pgap[:,self.mapping], leaf_text, parent_text[:,self.mapping]),-1)
        parent_input = torch.stack((pf,pgap,cf,child_second,cf-child_second,text_best.clamp(min=-1.0),
                                   parent_text,pf-cf),-1)
        q1,q2 = self.top(parent_best)
        l1,l2 = self.top(f1)
        root_input = torch.stack((q1,q2,q1-q2,l1,l2,l1-l2,q1-l1,parent_text.max(1).values),1)
        if self.experts is not None:
            ids = torch.arange(self.p,device=z.device)
            pe = self.experts(ids)[None].expand(len(z),-1,-1)
            parent_input = torch.cat((parent_input,pe),-1)
            leaf_input = torch.cat((leaf_input,pe[:,self.mapping]),-1)
        leaf = self.leaf_head(leaf_input).squeeze(-1).masked_fill(~active,-10000.0)
        parent_unknown = self.parent_head(parent_input).squeeze(-1).masked_fill(~present,-10000.0)
        root = self.root_head(root_input)
        return torch.cat((leaf,parent_unknown,root),1), z

    def adversarial_logits(self, fine):
        return self.parent_adversary(Reverse.apply(self.adapt(fine)))
