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
    hist = {"epoch": [], "train_obj": [], "train_mae": [], "val": []}

    it = range(1, config.max_epochs + 1)
    if show_epoch_bar:
        it = tqdm(it, desc=progress_desc, leave=leave_bar)

    for epoch in it:
        model.train()
        tot_obj = 0.0   # spike-aware + Laplacian (the optimized objective)
        tot_mae = 0.0   # plain MAE (comparable to validation)

        for batch in loaders["train"]:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["mask"].to(device)

            # ---- sanitize inputs (avoid NaN/Inf entering the model) ----
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

            opt.zero_grad()
            yhat = model(x)
            yhat = torch.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)

            # 1) objective used for optimization
            obj = masked_mae(yhat, y, m, activity_gamma=4.0, activity_mode="binary")
            if getattr(config, "lambda_lap", 0.0) > 0.0:
                if A_bin is None:
                    raise ValueError("lambda_lap > 0 but A_bin=None.")
                obj = obj + config.lambda_lap * laplacian_smoothness(yhat, A_bin)

            # 2) plain MAE for logging (matches how we compute validation)
            mae_plain = masked_mae(yhat, y, m)

            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tot_obj += obj.item()
            tot_mae += mae_plain.item()

        train_obj = tot_obj / max(1, len(loaders["train"]))
        train_mae = tot_mae / max(1, len(loaders["train"]))

        model.eval()
        with torch.no_grad():
            val_mae = 0.0
            for batch in loaders["val"]:
                x = batch["x"].to(device); y = batch["y"].to(device); m = batch["mask"].to(device)
                yhat = model(x)
                yhat = torch.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)
                val_mae += masked_mae(yhat, y, m).item()
            val_mae /= max(1, len(loaders["val"]))

        hist["epoch"].append(epoch)
        hist["train_obj"].append(train_obj)
        hist["train_mae"].append(train_mae)
        hist["val"].append(val_mae)

        if epoch_hook is not None:
            epoch_hook(epoch, model, loaders)

        # optional live postfix on the epoch bar
        if show_epoch_bar:
            try:
                # show the comparable quantities
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
    if "train" not in model.history:
        model.history["train"] = model.history["train_mae"]
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

    # def epoch_hook(epoch, model, loaders_dict):
    #     Xhat = _rollout_preds_over_timeline(model, X, M, current_exog_container["arr"],
    #                                         cfg.H_in, cfg.H_out, cfg.device)
    #     exog_new = features.add_exogenous(
    #         times=times, E=E, config=cfg, weather_df=weather_df, X=X, M=M, A_bin=A_bin,
    #         use_graph_fill=True, lags_self=lags_self, lags_neigh=lags_neigh,
    #         ema_alpha=ema_alpha, fill_weights=fill_weights, Xhat_prev=Xhat)
    #     current_exog_container["arr"] = exog_new

    #     # gap-aware rebuild of loaders for train/val
    #     new_loaders = make_loaders_gapaware(times, X, M, exog_new, cfg, slices, shuffle_train=True)
    #     loaders_dict["train"] = new_loaders["train"]
    #     loaders_dict["val"]   = new_loaders["val"]

    def epoch_hook(epoch, model, loaders_dict):
        Xhat = _rollout_preds_over_timeline(model, X, M, current_exog_container["arr"],
                                            cfg.H_in, cfg.H_out, cfg.device)
        exog_new = features.add_exogenous(
            times=times, E=E, config=cfg, weather_df=weather_df, X=X, M=M, A_bin=A_bin,
            use_graph_fill=True, lags_self=lags_self, lags_neigh=lags_neigh,
            ema_alpha=ema_alpha, fill_weights=fill_weights, Xhat_prev=Xhat
        ).astype("float32")
        exog_new = np.nan_to_num(exog_new, nan=0.0, posinf=0.0, neginf=0.0)   # <— sanitize
        current_exog_container["arr"] = exog_new

        # ONLY rebuild TRAIN; keep VAL fixed
        new_train = make_loaders_gapaware(times, X, M, exog_new, cfg, {"train": slices["train"],
                                                                       "val": slices["train"],   # dummy
                                                                       "test": slices["train"]}, # dummy
                                          shuffle_train=True)["train"]
        loaders_dict["train"] = new_train

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

# def fit_linear_regressor(X_hist: np.ndarray, Y_fut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Per-edge y0 ≈ a * mean(x_hist) + b, solved by least squares (edge-wise).
#     X_hist: [B,H_in,E,F] (we use the main flow channel at [:,:,:,0])
#     Y_fut : [B,H_out,E]  (we fit to horizon-0)
#     """
#     Xh = X_hist[..., 0]     # [B,H_in,E]
#     y0 = Y_fut[:, 0, :]     # [B,E]
#     B, H_in, E = Xh.shape
#     mu = Xh.mean(axis=1)    # [B,E]
#     a = np.zeros(E, np.float32); b = np.zeros(E, np.float32)
#     for e in range(E):
#         Xd = np.column_stack([mu[:, e], np.ones(B, np.float32)])
#         coef, *_ = np.linalg.lstsq(Xd, y0[:, e], rcond=None)
#         a[e], b[e] = coef.astype(np.float32)
#     return a, b

# def predict_linear_regressor(a: np.ndarray, b: np.ndarray, X_hist: np.ndarray) -> np.ndarray:
#     mu = X_hist[..., 0].mean(axis=1)   # [B,E]
#     y0 = (mu * a[None, :]) + b[None, :]
#     return y0[:, None, :]              # [B,1,E] (repeat to H_out as needed)

# put in the notebook (or into src/training.py if you prefer)
# def fit_linear_regressor(ds_train) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Edge-wise linear regressor using windowed inputs with EXOGENEOUS features.
#     We build a design vector z by averaging exog channels over H_in.
#     If you want to include past-flow as well, set include_flow=True below.
#     Returns (W, b) where:
#       W: [E, Fz], b: [E]
#     """
#     include_flow = True  # set True to add mean past-flow as a feature

#     Z_list, Y_list, E_ref = [], [], None
#     for i in range(len(ds_train)):
#         b = ds_train[i]
#         x = b["x"].numpy()      # [H_in, E, F_in]  (channel 0 is main flow; 1.. are exog)
#         y = b["y"].numpy()      # [H_out, E]
#         m = b["mask"].numpy()   # [H_out, E]
#         if E_ref is None:
#             E_ref = x.shape[1]
#         # build feature vector per edge: mean over time of exog channels
#         if x.shape[2] > 1:
#             ex = x[..., 1:]                # drop main flow channel
#             z = ex.mean(axis=0)            # [E, F_exog]
#         else:
#             z = np.zeros((x.shape[1], 0), np.float32)
#         if include_flow:
#             mu_flow = x[..., 0].mean(axis=0)[..., None]  # [E,1]
#             z = np.concatenate([mu_flow, z], axis=1)     # [E, Fz]
#         # target is horizon-0 with mask==1
#         y0 = y[0]    # [E]
#         m0 = (m[0] > 0.5)  # [E] boolean
#         Z_list.append((z, m0))
#         Y_list.append(y0)

#     # Stack by concatenating edges across batches where mask==1
#     Z_blocks, y_blocks = [], []
#     for (z, m0), y0 in zip(Z_list, Y_list):
#         if m0.any():
#             Z_blocks.append(z[m0])      # [Ne,Fz]
#             y_blocks.append(y0[m0])     # [Ne]
#     if len(Z_blocks) == 0:
#         # degenerate: no valid rows; fall back to zeros
#         Fz = (1 if include_flow else 0) + max(0, Z_list[0][0].shape[1])
#         return np.zeros((E_ref, Fz), np.float32), np.zeros((E_ref,), np.float32)

#     Z_all = np.vstack(Z_blocks)   # [N,Fz]
#     y_all = np.concatenate(y_blocks, axis=0).astype(np.float32)  # [N]

#     # Solve a single global linear model per edge? No—fit per edge as in old code:
#     # Build W,b edge-wise using masked rows collected above per edge.
#     # To keep it simple and robust, solve a shared W,b across edges:
#     #   y ≈ Z @ w + b   (shared across edges)
#     A = np.column_stack([Z_all, np.ones((Z_all.shape[0], 1), np.float32)])
#     coef, *_ = np.linalg.lstsq(A, y_all, rcond=None)  # [Fz+1]
#     w_shared = coef[:-1].astype(np.float32)           # [Fz]
#     b_shared = float(coef[-1])

#     # Expand to per-edge parameters (same weights for all edges; simple baseline)
#     W = np.tile(w_shared[None, :], (E_ref, 1))        # [E,Fz]
#     b = np.full((E_ref,), b_shared, np.float32)
#     return W, b

# def predict_linear_regressor(W: np.ndarray, b: np.ndarray, ds_test) -> np.ndarray:
#     """
#     Predict horizon-0 with an edge-wise linear model:
#       - W: [E, Fz], b: [E]
#       - z is built by averaging exogenous channels over H_in for each edge.
#     Returns [B,1,E].
#     """
#     preds = []
#     for i in range(len(ds_test)):
#         it = ds_test[i]
#         x = it["x"].numpy()      # [H_in, E, F_in]
#         E = x.shape[1]

#         # exogenous features = channels 1.. ; average over H_in -> [E, F_exog]
#         if x.shape[2] > 1:
#             ex = x[..., 1:]
#             z = ex.mean(axis=0)  # [E, Fz]
#         else:
#             # no exog present; use zeros of the same feature width as W
#             z = np.zeros((E, W.shape[1]), dtype=np.float32)

#         # PER-EDGE dot product (not a cross-edge @)
#         y0_e = (z * W).sum(axis=1) + b   # [E]

#         preds.append(y0_e[None, :])      # [1, E]
#     return np.stack(preds, axis=0)       # [B, 1, E]

def fit_linear_regressor(ds_train, include_flow: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Edge-wise linear regressor using windowed inputs with EXOG features.
    Features z = [ (optional mean past-flow), mean(exog over H_in) ] per edge.
    Returns:
      W: [E, Fz], b: [E]
    """
    Z_list, Y_list, E_ref = [], [], None
    for i in range(len(ds_train)):
        b = ds_train[i]
        x = b["x"].numpy()      # [H_in, E, F_in]  (channel 0 = main flow; 1.. = exog)
        y = b["y"].numpy()      # [H_out, E]
        m = b["mask"].numpy()   # [H_out, E]
        if E_ref is None:
            E_ref = x.shape[1]

        # exogenous mean over H_in
        if x.shape[2] > 1:
            ex = x[..., 1:]
            z = ex.mean(axis=0)              # [E, F_exog]
        else:
            z = np.zeros((x.shape[1], 0), np.float32)

        # optional mean past-flow
        if include_flow:
            mu_flow = x[..., 0].mean(axis=0)[..., None]  # [E,1]
            z = np.concatenate([mu_flow, z], axis=1)     # [E, Fz]

        y0 = y[0]                  # [E]
        m0 = (m[0] > 0.5)          # [E] boolean
        Z_list.append((z, m0))
        Y_list.append(y0)

    # stack only masked rows
    Z_blocks, y_blocks = [], []
    for (z, m0), y0 in zip(Z_list, Y_list):
        if m0.any():
            Z_blocks.append(z[m0])        # [Ne, Fz]
            y_blocks.append(y0[m0])       # [Ne]
    if len(Z_blocks) == 0:
        Fz = (1 if include_flow else 0) + (Z_list[0][0].shape[1] if Z_list else 0)
        return np.zeros((E_ref, Fz), np.float32), np.zeros((E_ref,), np.float32)

    Z_all = np.vstack(Z_blocks).astype(np.float32)   # [N, Fz]
    y_all = np.concatenate(y_blocks, axis=0).astype(np.float32)  # [N]

    # shared weights (simple, robust)
    A = np.column_stack([Z_all, np.ones((Z_all.shape[0], 1), np.float32)])
    coef, *_ = np.linalg.lstsq(A, y_all, rcond=None)  # [Fz+1]
    w_shared = coef[:-1].astype(np.float32)           # [Fz]
    b_shared = float(coef[-1])

    W = np.tile(w_shared[None, :], (E_ref, 1))        # [E, Fz]
    b = np.full((E_ref,), b_shared, np.float32)
    return W, b


def predict_linear_regressor(W: np.ndarray, b: np.ndarray, ds_test, include_flow: bool = True) -> np.ndarray:
    """
    Build z for each test window exactly like in fit(), then align width to W.shape[1].
    Returns [B, 1, E] (horizon-0 predictions).
    """
    Fz = W.shape[1]
    preds = []
    for i in range(len(ds_test)):
        it = ds_test[i]
        x = it["x"].numpy()    # [H_in, E, F_in]
        E  = x.shape[1]

        # exog mean
        if x.shape[2] > 1:
            ex = x[..., 1:]
            z = ex.mean(axis=0)               # [E, F_exog]
        else:
            z = np.zeros((E, 0), np.float32)

        # optional flow mean
        if include_flow:
            mu_flow = x[..., 0].mean(axis=0)[..., None]  # [E,1]
            z = np.concatenate([mu_flow, z], axis=1)     # [E, ?]

        # align to Fz expected by W
        if z.shape[1] < Fz:
            pad = np.zeros((E, Fz - z.shape[1]), np.float32)
            z = np.concatenate([z, pad], axis=1)
        elif z.shape[1] > Fz:
            z = z[:, :Fz]

        y0_e = (z * W).sum(axis=1) + b   # [E]
        preds.append(y0_e[None, :])      # [1, E]

    return np.stack(preds, axis=0)       # [B, 1, E]


# =========================
#   HISTORICAL AVERAGE BASELINE
# =========================

# def hist_avg_predict_masked(X: np.ndarray,
#                             M: np.ndarray,
#                             times: pd.DatetimeIndex,
#                             masks_final: dict,
#                             H_in: int, H_out: int,
#                             *,
#                             ds_test,
#                             use_dow: bool = False) -> np.ndarray:
#     """
#     Historical-average baseline with mask:
#       - use_dow=False: mean per time-of-day
#       - use_dow=True : mean per (day-of-week, time-of-day)
#     Returns [B, H_out, E] aligned to loaders_final["test"].
#     """
#     idx_fit = np.where(masks_final["train"])[0]
#     t_fit = pd.DatetimeIndex(times[idx_fit])
#     tod_fit = (t_fit.hour*3600 + t_fit.minute*60 + t_fit.second).to_numpy()
#     if use_dow:
#         dow_fit = t_fit.dayofweek.to_numpy()

#     E = X.shape[1]
#     means = {}  # key -> [E]
#     if use_dow:
#         keys = np.stack([dow_fit, tod_fit], axis=1)
#     else:
#         keys = tod_fit[:, None]

#     for k in np.unique(keys, axis=0):
#         if use_dow:
#             sel = np.where((dow_fit == k[0]) & (tod_fit == k[1]))[0]
#             key = (int(k[0]), int(k[1]))
#         else:
#             sel = np.where(tod_fit == k[0])[0]
#             key = (int(k[0]),)
#         if len(sel) == 0:
#             continue
#         rows = idx_fit[sel]
#         W = M[rows, :]                     # [n,E]
#         num = (X[rows, :] * W).sum(axis=0) # [E]
#         den = W.sum(axis=0).clip(min=1.0)  # [E]
#         mu  = (num / den).astype(np.float32)
#         means[key] = mu

#     # global fallback (masked overall mean on train rows)
#     Wg = M[idx_fit, :]
#     fallback = ((X[idx_fit, :]*Wg).sum(axis=0) / Wg.sum(axis=0).clip(min=1.0)).astype(np.float32)

#     # Roll out for test windows
#     preds = []
#     for i in range(len(ds_test)):
#         t0 = int(ds_test.starts[i]); t1 = t0 + H_in
#         ts = pd.DatetimeIndex(times[t1:t1+H_out])
#         out = np.zeros((H_out, E), np.float32)
#         for h, tt in enumerate(ts):
#             if use_dow:
#                 k = (int(tt.dayofweek), int(tt.hour*3600 + tt.minute*60 + tt.second))
#             else:
#                 k = (int(tt.hour*3600 + tt.minute*60 + tt.second),)
#             out[h] = means.get(k, fallback)
#         preds.append(out)
#     return np.stack(preds, axis=0)  # [B,H_out,E]

def hist_avg_predict_masked(
    X: np.ndarray,
    M: np.ndarray,
    times: pd.DatetimeIndex,
    masks_final: dict,
    H_in: int, H_out: int,
    *,
    ds_test,
    bin_minutes: int | None = None,   # coarser = weaker baseline; set None to use exact minute
) -> np.ndarray:
    """
    Naive time-of-day mean baseline (no mask, no cleanup):
      - Compute per-edge mean over TRAIN rows for each time-of-day bucket.
      - No use of M; missing/zeros in X affect the mean as-is.
      - Optionally coarsen TOD into `bin_minutes` buckets (e.g., 60 min).

    Returns [B, H_out, E], aligned to ds_test windows.
    """
    idx_fit = np.where(masks_final["train"])[0]
    t_fit = pd.DatetimeIndex(times[idx_fit])
    E = X.shape[1]

    # time-of-day seconds
    tod_sec = (t_fit.hour * 3600 + t_fit.minute * 60 + t_fit.second).astype(int).to_numpy()

    if bin_minutes is not None and bin_minutes > 0:
        bin_size = bin_minutes * 60
        tod_key = (tod_sec // bin_size) * bin_size
    else:
        tod_key = tod_sec

    # per-bucket naive mean (no mask)
    means = {}
    for k in np.unique(tod_key):
        sel = np.where(tod_key == k)[0]
        rows = idx_fit[sel]
        if rows.size == 0:
            continue
        mu = X[rows, :].mean(axis=0).astype(np.float32)   # <-- naive mean, no mask
        means[int(k)] = mu

    # fallback = global naive mean on TRAIN rows
    fallback = X[idx_fit, :].mean(axis=0).astype(np.float32)

    # roll out for test windows
    preds = []
    for i in range(len(ds_test)):
        t0 = int(ds_test.starts[i]); t1 = t0 + H_in
        ts = pd.DatetimeIndex(times[t1:t1+H_out])
        out = np.zeros((H_out, E), np.float32)
        for h, tt in enumerate(ts):
            sec = tt.hour * 3600 + tt.minute * 60 + tt.second
            key = ((sec // (bin_minutes*60)) * (bin_minutes*60)) if (bin_minutes and bin_minutes>0) else sec
            out[h] = means.get(int(key), fallback)
        preds.append(out)
    return np.stack(preds, axis=0)
