from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from scripts.features import sanitize_and_row_normalize_A


class GCNLayer(nn.Module):
    def __init__(self, F_in: int, F_out: int, A_hat: torch.Tensor, dropout=0.0):
        super().__init__()
        # make sure dtype is float32 at registration time
        self.register_buffer('A_hat', A_hat.to(dtype=torch.float32))
        self.lin = nn.Linear(F_in, F_out, bias=True)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, X):  # X: [B,T,E,F_in]
        # ensure the buffer lives on the same device as activations
        A = self.A_hat.to(X.device, non_blocking=True)   # <— important
        Xw = self.lin(X)                                 # [B,T,E,F_out]
        Xprop = torch.einsum('ij,btjf->btif', A, Xw)     # [B,T,E,F_out]
        return self.do(self.act(Xprop))

class ChebGCN(nn.Module):
    """
    Chebyshev-style K-hop graph convolution on the line-graph of edges.
    Uses pre-normalized adjacency (A_hat) and builds multi-hop responses.
    """
    def __init__(self, F_in, F_out, A_hat: torch.Tensor, K=3, dropout=0.1):
        super().__init__()
        self.register_buffer('A_hat', A_hat.to(dtype=torch.float32))  # [E,E]
        self.W = nn.Parameter(torch.randn(K, F_in, F_out) * 0.01)
        self.act = nn.ReLU()
        self.do  = nn.Dropout(dropout)
        self.K = K

    def forward(self, X):  # X: [B,T,E,F_in]
        A = self.A_hat.to(X.device, non_blocking=True)  # [E,E]
        B,T,E,F_in = X.shape
        outs = []

        # k = 0  (identity / 0-hop)
        Z = X  # [B,T,E,F_in]
        outs.append(torch.einsum('btif,kfo->btio', Z, self.W[0:1])[...,0,:])  # apply W0

        # iterative k-hop aggregations: Z <- A @ Z
        cur = Z
        for k in range(1, self.K):
            cur = torch.einsum('ij,btjf->btif', A, cur)                    # A @ cur
            outs.append(torch.einsum('btif,kfo->btio', cur, self.W[k:k+1])[...,0,:])

        Y = torch.stack(outs, dim=0).sum(0)  # sum_k (A^k X) W_k   -> [B,T,E,F_out]
        return self.do(self.act(Y))


class TemporalConv(nn.Module):
    def __init__(self, F_in: int, F_out: int, k: int=3, dilation: int=1, dropout=0.0):
        super().__init__()
        padding = (k-1)*dilation
        self.conv = nn.Conv2d(in_channels=F_in, out_channels=F_out,
                              kernel_size=(k,1), dilation=(dilation,1), padding=(padding,0))
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)
    def forward(self, X):  # X: [B,T,E,F] -> make channels=F, height=T, width=E
        B,T,E,F = X.shape
        Xp = X.permute(0,3,1,2).contiguous()     # [B,F,T,E]
        Y = self.conv(Xp)                      # causal-ish with left padding
        Y = Y[:,:,:T,:]                        # trim to keep length
        Y = self.act(Y)
        Y = self.do(Y)
        return Y.permute(0,2,3,1)             # [B,T,E,F_out]


class STBlock(nn.Module):
    def __init__(self, F_in: int, F_hidden: int, A_hat: torch.Tensor,
                 dropout=0.1, dilation=1, gcn_type="gcn", cheb_K=3):
        super().__init__()
        self.temp1 = TemporalConv(F_in, F_hidden, k=3, dilation=dilation, dropout=dropout)
        if gcn_type == "cheb":
            self.gcn = ChebGCN(F_hidden, F_hidden, A_hat, K=cheb_K, dropout=dropout)
        else:
            self.gcn = GCNLayer(F_hidden, F_hidden, A_hat, dropout=dropout)
        self.temp2 = TemporalConv(F_hidden, F_hidden, k=3, dilation=1, dropout=dropout)
        self.resid = nn.Conv2d(F_in, F_hidden, kernel_size=(1,1))
    def forward(self, X):  # [B,T,E,F_in]
        R = self.resid(X.permute(0,3,1,2))    # [B,F_hidden,T,E]
        R = R.permute(0,2,3,1)
        out = self.temp1(X)
        out = self.gcn(out)
        out = self.temp2(out)
        return out + R


class EdgeSTGNN(nn.Module):
    def __init__(self, E: int, F_in: int, H_out: int, A_hat: np.ndarray,
                 nblocks=2, hidden=32, dropout=0.1,
                 gcn_type="gcn", cheb_K=3):
        super().__init__()
        A_hat = sanitize_and_row_normalize_A(A_hat)
        A = torch.from_numpy(A_hat).to(torch.float32)
        self.register_buffer('A_hat', A)

        blocks = []
        dil = 1
        Fin = F_in
        for _ in range(nblocks):
            blocks.append(STBlock(Fin, hidden, self.A_hat,
                                  dropout=dropout, dilation=dil,
                                  gcn_type=gcn_type, cheb_K=cheb_K))
            Fin = hidden
            dil *= 2                    # 1,2,4,...
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(hidden, H_out, kernel_size=(1,1))
        )

    def forward(self, X):  # [B,T_in,E,F_in]
        out = X
        for blk in self.blocks:
            out = blk(out)
        out = out.permute(0,3,1,2)
        out = self.head(out)             # [B,H_out,T,E]
        out = out[:,:,-1,:]              # [B,H_out,E]
        out = nn.functional.softplus(out)
        return out

    

# # to do!!!

# # ChebGCN(F_hidden, F_hidden, A_hat, K=3)
# class ChebGCN(nn.Module):
#     def __init__(self, F_in, F_out, A_hat: torch.Tensor, K=3, dropout=0.1):
#         super().__init__()
#         self.register_buffer('A', A_hat)           # [E,E], row-normalized with self-loops
#         self.W = nn.Parameter(torch.randn(K, F_in, F_out) * 0.01)
#         self.do = nn.Dropout(dropout)
#         self.act = nn.ReLU()
#         self.K = K

#     def forward(self, X):  # X: [B,T,E,F_in]
#         B,T,E,F = X.shape
#         A = self.A                                           # [E,E]
#         Xw = torch.einsum('bt ef, k f o -> bt k e o', X, self.W)  # [B,T,K,E,F_out]
#         # compute powers A^k @ X for k=0..K-1 (k=0 is identity)
#         outs = []
#         cur = Xw[:, :, 0]                                    # k=0
#         outs.append(cur)
#         for k in range(1, self.K):
#             cur = torch.einsum('ij,btjf->btif', A, Xw[:, :, k])
#             outs.append(cur)
#         Y = sum(outs)                                        # [B,T,E,F_out]
#         return self.do(self.act(Y))

# # include Neighbor lag features (lags 1–3)!!!!!!!!!!

# # extra loss term
# def laplacian_smoothness(yhat, A_bin):  # yhat: [B,H,E]
#     # sum over neighbors (i,j) of (y_i - y_j)^2; normalize by number of pairs
#     diff = yhat[..., None, :] - yhat[..., :, None]         # [B,H,E,E]
#     mask = torch.from_numpy(A_bin).to(yhat.device).bool()
#     sq = (diff**2)[..., mask]                               # select neighbors
#     return sq.mean()



# blocks = nn.ModuleList([
#     STBlock(F_in, 64, A_hat, dropout=0.12, dilation=1),
#     STBlock(64,   64, A_hat, dropout=0.12, dilation=2),
#     STBlock(64,   64, A_hat, dropout=0.12, dilation=4),
# ])


