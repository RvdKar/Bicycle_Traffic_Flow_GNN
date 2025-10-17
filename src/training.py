from __future__ import annotations
import math
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import preprocessing, features
from .models import EdgeSTGNN

# =========================
#        LOSSES
# =========================

def spike_aware_loss(pred, target, mask, tau, alpha=1.5, beta=0.3):
    w = 1.0 + alpha * (target / tau).clamp(min=0)
    base = (pred - target).abs()
    num = (base * w * mask).sum()
    den = (w * mask).sum().clamp_min(1.0)
    loss_base = num / den
    d_pred = pred[:, 1:, :] - pred[:, :-1, :]
    d_tgt  = target[:, 1:, :] - target[:, :-1, :]
    d_mask = mask[:, 1:, :] * mask[:, :-1, :]
    loss_slope = (d_mask * (d_pred - d_tgt).abs()).sum() / d_mask.sum().clamp_min(1.0)
    return loss_base + beta * loss_slope

def masked_mae(pred, target, mask, eps=1e-8, *,
               activity_gamma: float = 0.0, activity_mode: str = "binary",
               activity_q: float | None = None, activity_wmax: float = 3.0):
    pred   = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    m      = mask.to(dtype=torch.float32)
    if activity_gamma > 0.0:
        if activity_mode == "binary":
            w = 1.0 + activity_gamma * (target > 0).float()
        elif activity_mode == "value":
            if activity_q is None:
                obs = target[m > 0]
                q = torch.quantile(obs, 0.95) if obs.numel() > 0 else torch.tensor(1.0, device=pred.device)
            else:
                q = torch.as_tensor(activity_q, dtype=torch.float32, device=pred.device)
            w_raw = (target / (q + 1e-12)).clamp(min=0.0, max=activity_wmax)
            w = 1.0 + activity_gamma * w_raw
            w = torch.where(target > 0, w, torch.ones_like(w))
        else:
            raise ValueError("activity_mode must be 'binary' or 'value'")
    else:
        w = torch.ones_like(pred)
    diff = (pred - target).abs()
    num = (diff * m * w).sum(); den = (m * w).sum().clamp_min(eps)
    return num / den

def masked_mse(pred, target, mask, eps=1e-8):
    m = (mask > 0).to(pred.dtype)
    diff2 = (pred - target) ** 2
    num = (diff2 * m).sum(); den = m.sum().clamp_min(1.0)
    return num / den

# =========================
#   REGULARISER (OPTIONAL)
# =========================

def laplacian_smoothness(yhat: torch.Tensor, A_bin_np: np.ndarray) -> torch.Tensor:
    """
    Simple smoothness penalty over edges in line-graph.
    yhat: [B, H_out, E]
    """
    A = torch.from_numpy(A_bin_np).to(yhat.device, non_blocking=True).bool()
    diff = yhat[..., None, :] - yhat[..., :, None]
    sq = (diff**2)[..., A]
    return sq.mean()

# =========================
#        TRAIN / EVAL
# =========================

from tqdm.auto import tqdm  # make sure this is at the top of training.py

def train_model(model: nn.Module,
                loaders: Dict[str, DataLoader],
                config: preprocessing.Config,
                epoch_hook=None,
                A_bin: np.ndarray | None = None,
                *,
                show_epoch_bar: bool = True,
                leave_bar: bool = False,
                progress_desc: str = "Epochs"):
    """
    Train with early stopping (val MAE). Progress bar can be enabled/disabled per call.
    """
    device = config.device
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    patience = config.patience
    hist = {"epoch": [], "train": [], "val": []}

    it = range(1, config.max_epochs + 1)
    if show_epoch_bar:
        it = tqdm(it, desc=progress_desc, leave=leave_bar)

    for epoch in it:
        model.train()
        tot = 0.0
        for batch in loaders["train"]:
            x = batch["x"].to(device)         # [B,H_in,E,F_in]
            y = batch["y"].to(device)         # [B,H_out,E]
            m = batch["mask"].to(device)      # [B,H_out,E]
            opt.zero_grad()
            yhat = model(x)                   # [B,H_out,E]
            loss = masked_mae(yhat, y, m, activity_gamma=4.0, activity_mode="binary")
            if getattr(config, "lambda_lap", 0.0) > 0.0:
                if A_bin is None:
                    raise ValueError("lambda_lap > 0 but A_bin=None.")
                loss = loss + config.lambda_lap * laplacian_smoothness(yhat, A_bin)
            loss.backward()
            opt.step()
            tot += loss.item()
        train_mae = tot / max(1, len(loaders["train"]))

        model.eval()
        with torch.no_grad():
            val_mae = 0.0
            for batch in loaders["val"]:
                x = batch["x"].to(device); y = batch["y"].to(device); m = batch["mask"].to(device)
                yhat = model(x)
                val_mae += masked_mae(yhat, y, m).item()
            val_mae /= max(1, len(loaders["val"]))

        hist["epoch"].append(epoch); hist["train"].append(train_mae); hist["val"].append(val_mae)

        if epoch_hook is not None:
            epoch_hook(epoch, model, loaders)

        # optional live postfix on the epoch bar
        if show_epoch_bar:
            try:
                it.set_postfix(train=f"{train_mae:.3f}", val=f"{val_mae:.3f}", best=f"{best['val']:.3f}")
            except Exception:
                pass

        if val_mae < best["val"] - 1e-6:
            best = {"val": val_mae, "state": model.state_dict(), "epoch": epoch}
            patience = config.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.history = hist
    return model

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str,float]:
    model.eval(); mae_sum = rmse_sum = m_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device); y = batch["y"].to(device); m = batch["mask"].to(device)
            yhat = model(x); err = (yhat - y)
            mae_sum += (err.abs() * m).sum().item()
            rmse_sum += ((err**2) * m).sum().item()
            m_sum += m.sum().item()
    mae = mae_sum / (m_sum + 1e-8); rmse = math.sqrt(rmse_sum / (m_sum + 1e-8))
    return {"MAE": mae, "RMSE": rmse}

# =========================
#  PREDICTION-DRIVEN HOOK
# =========================

def _rollout_preds_over_timeline(model, X, M, exog, H_in, H_out, device):
    """Greedy horizon-0 rollout over the full timeline to refresh exog with predictions."""
    model.eval()
    T, E = X.shape
    Xhat = np.zeros((T, E), np.float32)
    ds_all = preprocessing.WindowedEdgeDataset(X, M, exog, H_in, H_out)
    loader_all = DataLoader(ds_all, batch_size=128, shuffle=False, drop_last=False,
                            pin_memory=str(device).startswith("cuda"))
    t0 = H_in; offset = 0
    with torch.no_grad():
        for batch in loader_all:
            xb = batch["x"].to(device)
            yb = model(xb).cpu().numpy()  # [B,H_out,E]
            B = yb.shape[0]
            Xhat[t0 + offset : t0 + offset + B, :] = yb[:, 0, :]
            offset += B
    return Xhat

def make_prediction_driven_hook(X, M, times, E, cfg, weather_df, A_bin, slices, loaders,
                                *, lags_self=(1,2,3,6,12), lags_neigh=(1,2,3),
                                ema_alpha=0.7, fill_weights=(0.5,0.3,0.2),
                                current_exog_container=None):
    """Rebuild exogenous each epoch using prior predictions; updates loaders in-place."""
    if current_exog_container is None:
        current_exog_container = {"arr": features.add_exogenous(
            times=times, E=E, config=cfg, weather_df=weather_df, X=X, M=M, A_bin=A_bin,
            use_graph_fill=True, lags_self=lags_self, lags_neigh=lags_neigh,
            ema_alpha=ema_alpha, fill_weights=(0.7,0.3,0.0), Xhat_prev=None)}

    def epoch_hook(epoch, model, loaders_dict):
        Xhat = _rollout_preds_over_timeline(model, X, M, current_exog_container["arr"],
                                            cfg.H_in, cfg.H_out, cfg.device)
        exog_new = features.add_exogenous(
            times=times, E=E, config=cfg, weather_df=weather_df, X=X, M=M, A_bin=A_bin,
            use_graph_fill=True, lags_self=lags_self, lags_neigh=lags_neigh,
            ema_alpha=ema_alpha, fill_weights=fill_weights, Xhat_prev=Xhat)
        current_exog_container["arr"] = exog_new

        # gap-aware rebuild of loaders for train/val
        new_loaders = make_loaders_gapaware(times, X, M, exog_new, cfg, slices, shuffle_train=True)
        loaders_dict["train"] = new_loaders["train"]
        loaders_dict["val"]   = new_loaders["val"]

    return epoch_hook

# =========================
#   GAP-AWARE LOADERS
# =========================

def compute_gap_aware_starts(times: pd.DatetimeIndex,
                             mask_bool: np.ndarray,
                             H_in: int, H_out: int,
                             granularity: str) -> list[int]:
    """
    Valid start indices s on the FULL timeline such that:
      - window [s, s+H_in+H_out) is entirely inside the split mask; and
      - timestamps are contiguous at `granularity` (no gaps).
    """
    times = pd.DatetimeIndex(times)
    dt = pd.Timedelta(granularity)
    L = H_in + H_out

    idx = np.where(mask_bool)[0]
    if idx.size == 0:
        return []

    # find contiguous runs within the masked indices
    runs = []
    run_start = 0
    for i in range(len(idx) - 1):
        if (times[idx[i + 1]] - times[idx[i]]) != dt:
            runs.append((idx[run_start], idx[i] + 1))  # [global_start, global_end)
            run_start = i + 1
    runs.append((idx[run_start], idx[-1] + 1))

    # within each run, all starts whose full window fits inside the run
    starts = []
    for gs, ge in runs:
        if ge - gs >= L:
            starts.extend(range(gs, ge - L + 1))
    return starts

def make_loaders_gapaware(times: pd.DatetimeIndex,
                          X: np.ndarray, M: np.ndarray, exog: Optional[np.ndarray],
                          cfg: preprocessing.Config,
                          masks: Dict[str, np.ndarray],
                          shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """DataLoaders using gap-aware start indices per split on the full timeline."""
    def _to_bool(a):  # supports np.array or pd.Series
        return a.to_numpy(dtype=bool) if hasattr(a, "to_numpy") else a.astype(bool)

    starts = {
        split: compute_gap_aware_starts(times, _to_bool(masks[split]), cfg.H_in, cfg.H_out, cfg.time_granularity)
        for split in ("train", "val", "test")
    }
    ds = {
        split: preprocessing.WindowedEdgeDataset(X, M, (None if exog is None else exog),
                                                cfg.H_in, cfg.H_out,
                                                valid_start_indices=starts[split])
        for split in starts
    }
    loaders = {
        split: DataLoader(
            ds[split],
            batch_size=cfg.batch_size,
            shuffle=(shuffle_train and split == "train"),
            drop_last=(split == "train"),
            pin_memory=str(cfg.device).startswith("cuda"),
        )
        for split in ds
    }
    return loaders

# =========================
#      MODEL HELPERS
# =========================

def build_gnn(E: int, Fin: int, Hout: int, A_hat: np.ndarray, cfg: preprocessing.Config) -> EdgeSTGNN:
    return EdgeSTGNN(E=E, F_in=Fin, H_out=Hout, A_hat_np=A_hat,
                     nblocks=cfg.nblocks, hidden=cfg.hidden, dropout=cfg.dropout,
                     gcn_type=cfg.gcn_type, cheb_K=cfg.cheb_K)

# =========================
#   LINEAR BASELINE (tiny)
# =========================

def fit_linear_regressor(X_hist: np.ndarray, Y_fut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-edge y0 ≈ a * mean(x_hist) + b, solved by least squares (edge-wise).
    X_hist: [B,H_in,E,F] (we use the main flow channel at [:,:,:,0])
    Y_fut : [B,H_out,E]  (we fit to horizon-0)
    """
    Xh = X_hist[..., 0]     # [B,H_in,E]
    y0 = Y_fut[:, 0, :]     # [B,E]
    B, H_in, E = Xh.shape
    mu = Xh.mean(axis=1)    # [B,E]
    a = np.zeros(E, np.float32); b = np.zeros(E, np.float32)
    for e in range(E):
        Xd = np.column_stack([mu[:, e], np.ones(B, np.float32)])
        coef, *_ = np.linalg.lstsq(Xd, y0[:, e], rcond=None)
        a[e], b[e] = coef.astype(np.float32)
    return a, b

def predict_linear_regressor(a: np.ndarray, b: np.ndarray, X_hist: np.ndarray) -> np.ndarray:
    mu = X_hist[..., 0].mean(axis=1)   # [B,E]
    y0 = (mu * a[None, :]) + b[None, :]
    return y0[:, None, :]              # [B,1,E] (repeat to H_out as needed)
