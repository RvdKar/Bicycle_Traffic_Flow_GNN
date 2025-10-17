from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import math

__all__ = ["GCNLayer", "ChebGCN", "TemporalConv", "STBlock", "EdgeSTGNN"]

# ---------------- Basic GCN layer ----------------

class GCNLayer(nn.Module):
    def __init__(self, F_in: int, F_out: int, A_hat: torch.Tensor, dropout: float = 0.0):
        """
        A_hat must be a precomputed propagation matrix of shape [E, E] (row- or sym-normalized).
        X shape: [B, T, E, F_in]
        """
        super().__init__()
        self.register_buffer("A_hat", A_hat.to(dtype=torch.float32))
        self.lin = nn.Linear(F_in, F_out, bias=True)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, T, E, F_in]
        A = self.A_hat.to(X.device, non_blocking=True)
        Xw = self.lin(X)  # [B, T, E, F_out]
        Xprop = torch.einsum("ij,btjf->btif", A, Xw)  # graph mix on edge axis
        return self.do(self.act(Xprop))


# ---------------- "Chebyshev" K-hop diffusion (power-of-A) ----------------

class ChebGCN(nn.Module):
    """
    K-order polynomial GCN on normalized adjacency A_hat.
    Input:  X  [B,T,E,F_in]
    Output: Y  [B,T,E,F_out]
    """
    def __init__(self, F_in: int, F_out: int, A_hat: torch.Tensor, K: int = 3, dropout: float = 0.1):
        super().__init__()
        assert K >= 1
        self.K = int(K)
        # keep a finite, float32 copy on buffer
        A_hat = A_hat.detach().float().clone()
        A_hat[~torch.isfinite(A_hat)] = 0.0
        self.register_buffer("A_hat", A_hat)  # [E,E]

        # weights per hop
        W = torch.empty(self.K, F_in, F_out)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        self.W = nn.Parameter(W)  # [K, F_in, F_out]

        self.act = nn.ReLU()
        self.do  = nn.Dropout(dropout)

    def forward(self, X):  # X: [B,T,E,F_in]
        A = self.A_hat.to(X.device, non_blocking=True)  # [E,E]
        B,T,E,F_in = X.shape

        # k = 0 term (identity)
        Y = torch.einsum("btif,fo->btio", X, self.W[0])  # [B,T,E,F_out]

        # k >= 1 terms: iterative A @ X
        cur = X
        for k in range(1, self.K):
            # propagate once
            cur = torch.einsum("ij,btjf->btif", A, cur)   # [B,T,E,F_in]
            Y   = Y + torch.einsum("btif,fo->btio", cur, self.W[k])

        Y = self.act(Y)
        Y = self.do(Y)
        return Y



# ---------------- Temporal conv ----------------

class TemporalConv(nn.Module):
    def __init__(self, F_in: int, F_out: int, k: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv = nn.Conv2d(F_in, F_out, kernel_size=(k, 1), dilation=(dilation, 1), padding=(padding, 0))
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, T, E, F]
        B, T, E, F = X.shape
        Y = self.conv(X.permute(0, 3, 1, 2))  # [B, F_out, T', E]
        Y = Y[:, :, :T, :]  # trim to original T
        Y = self.do(self.act(Y))
        return Y.permute(0, 2, 3, 1)  # [B, T, E, F_out]


# ---------------- Spatio-temporal block ----------------

class STBlock(nn.Module):
    def __init__(
        self,
        F_in: int,
        F_hidden: int,
        A_hat: torch.Tensor,
        dropout: float = 0.1,
        dilation: int = 1,
        gcn_type: str = "gcn",
        cheb_K: int = 3,
    ):
        super().__init__()
        self.temp1 = TemporalConv(F_in, F_hidden, k=3, dilation=dilation, dropout=dropout)
        if gcn_type == "cheb":
            self.gcn = ChebGCN(F_hidden, F_hidden, A_hat, K=cheb_K, dropout=dropout)
        else:
            self.gcn = GCNLayer(F_hidden, F_hidden, A_hat, dropout=dropout)
        self.temp2 = TemporalConv(F_hidden, F_hidden, k=3, dilation=1, dropout=dropout)
        self.resid = nn.Conv2d(F_in, F_hidden, kernel_size=(1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, T, E, F_in]
        R = self.resid(X.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.temp1(X)
        out = self.gcn(out)
        out = self.temp2(out)
        return out + R


# ---------------- Model head ----------------

class EdgeSTGNN(nn.Module):
    def __init__(
        self,
        E: int,
        F_in: int,
        H_out: int,
        A_hat_np: np.ndarray,
        nblocks: int = 2,
        hidden: int = 32,
        dropout: float = 0.1,
        gcn_type: str = "gcn",
        cheb_K: int = 3,
    ):
        """
        A_hat_np must already be normalized and include self-loops exactly once.
        F_in = 1 + (#exogenous features)
        Input to forward: X [B, H_in, E, F_in]
        Output:           [B, H_out, E]
        """
        super().__init__()
        assert isinstance(A_hat_np, np.ndarray) and A_hat_np.shape == (E, E), "A_hat_np must be [E,E]"
        self.register_buffer("A_hat", torch.from_numpy(A_hat_np.astype(np.float32, copy=False)))

        blocks = []
        dil = 1
        Fin = F_in
        for _ in range(nblocks):
            blocks.append(
                STBlock(
                    Fin,
                    hidden,
                    self.A_hat,
                    dropout=dropout,
                    dilation=dil,
                    gcn_type=gcn_type,
                    cheb_K=cheb_K,
                )
            )
            Fin = hidden
            dil *= 2
        self.blocks = nn.ModuleList(blocks)

        # Head maps [B, T, E, hidden] -> [B, H_out, E]
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden, H_out, kernel_size=(1, 1)),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, H_in, E, F_in]
        out = X
        for blk in self.blocks:
            out = blk(out)  # [B, H_in, E, hidden]
        out = out.permute(0, 3, 1, 2)         # [B, hidden, H_in, E]
        out = self.head(out)                  # [B, H_out, H_in, E]
        out = out[:, :, -1, :]                # take last temporal step -> [B, H_out, E]
        return torch.nn.functional.softplus(out)
