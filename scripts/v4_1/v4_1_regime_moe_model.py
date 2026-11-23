"""A market-state-only, long-only three-expert router."""
import torch
from torch import nn


class RegimeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(5,12,3,padding=1),nn.SiLU(),
                                   nn.Conv1d(12,12,3,padding=1),nn.SiLU())
        self.router=nn.Sequential(nn.LayerNorm(24),nn.Dropout(.15),nn.Linear(24,3))

    def forward(self,states):
        h=self.encoder(states.transpose(1,2))
        pooled=torch.cat([h[:,:,-1],h.mean(2)],dim=1)
        return torch.softmax(self.router(pooled)/1.5,dim=1)


def ic_moments(experts,y):
    """Sufficient statistics for exact weekly Pearson on stocks with observed targets."""
    valid=torch.isfinite(y)
    count=valid.sum(1,keepdim=True).clamp_min(1)
    target=y.nan_to_num()
    yc=(target-(target*valid).sum(1,keepdim=True)/count)*valid
    xc=(experts-(experts*valid.unsqueeze(2)).sum(1,keepdim=True)/count.unsqueeze(2))*valid.unsqueeze(2)
    return xc.transpose(1,2)@xc,(xc*yc.unsqueeze(2)).sum(1),yc.square().sum(1)


def objective(w,gram,xy,yy,adjacent):
    var=(w.unsqueeze(1)@gram@w.unsqueeze(2)).flatten()
    ic=(w*xy).sum(1)/(var*yy).clamp_min(1e-16).sqrt()
    smooth=(w[1:]-w[:-1]).abs().sum(1)[adjacent].mean()
    prior=(w-1/3).square().sum(1).mean()
    return -ic.mean()/(ic.std(unbiased=False)+.05)+.15*smooth+.05*prior
