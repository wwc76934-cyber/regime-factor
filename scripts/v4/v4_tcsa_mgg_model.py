"""CPU temporal convolution, cross-stock attention and market-guided gates."""
import torch
from torch import nn


class TCSAMGG(nn.Module):
    def __init__(self, dim=24, attention=True, gating=True):
        super().__init__()
        branch = dim // 2
        self.local = nn.Sequential(nn.Conv1d(7,branch,3,padding=1),nn.SiLU(),
                                   nn.Conv1d(branch,branch,3,padding=2,dilation=2),nn.SiLU())
        self.wide = nn.Sequential(nn.Conv1d(7,branch,5,padding=2),nn.SiLU())
        self.temporal = nn.Sequential(nn.Linear(2*dim,dim),nn.LayerNorm(dim),nn.SiLU(),nn.Dropout(.2))
        self.attention = nn.MultiheadAttention(dim,4,dropout=.15,batch_first=True) if attention else None
        self.norm = nn.LayerNorm(dim)
        self.gating = nn.Sequential(nn.Linear(5,16),nn.SiLU(),nn.Linear(16,dim),nn.Sigmoid()) if gating else None
        self.head = nn.Sequential(nn.Linear(dim,16),nn.SiLU(),nn.Dropout(.2),nn.Linear(16,1))

    def forward(self, x, z, stock_mask):
        b,n,t,c = x.shape
        series = x.reshape(b*n,t,c).transpose(1,2)
        h = torch.cat([self.local(series),self.wide(series)],dim=1)
        h = self.temporal(torch.cat([h[:,:,-1],h.mean(-1)],dim=1)).reshape(b,n,-1)
        if self.attention is not None:
            attended,_ = self.attention(h,h,h,key_padding_mask=~stock_mask,need_weights=False)
            h = self.norm(h+attended)
        gate = (2*self.gating(z)) if self.gating is not None else torch.ones_like(h[:,0])
        return self.head(h*gate.unsqueeze(1)).squeeze(-1),gate


def ic_loss(scores, targets, valid):
    """Average weekly Pearson and MSE, keeping held-out or missing targets out of all moments."""
    count = valid.sum(-1,keepdim=True).clamp_min(1)
    y = targets.nan_to_num()
    def standardize(a):
        centered = (a-(a*valid).sum(-1,keepdim=True)/count)*valid
        return centered/((centered.square().sum(-1,keepdim=True)/count).clamp_min(1e-12).sqrt())
    p,zy = standardize(scores),standardize(y)
    corr = (p*zy).sum(-1,keepdim=True)/count
    mse = ((p-y).square()*valid).sum(-1,keepdim=True)/count
    return (-corr+.25*mse).mean()
