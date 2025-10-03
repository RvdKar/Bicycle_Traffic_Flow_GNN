# scripts/features.py
from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Iterable, Tuple

import random
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    # avoid runtime import / circular deps; only needed for type checking
    from preprocessing import Config


# ------------------------- utils -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def steps_for_hours(granularity: str, hours: int) -> int:
    step = pd.Timedelta(granularity)  # e.g., '5min' or '15min'
    return int(pd.Timedelta(f"{hours}h") / step)


def sanitize_and_row_normalize_A(A_hat_np: np.ndarray) -> np.ndarray:
    """Clean, add self-loops, and row-normalize adjacency."""
    A = np.array(A_hat_np, dtype=np.float32, copy=True)
    A[~np.isfinite(A)] = 0.0
    np.fill_diagonal(A, A.diagonal() + 1.0)
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < 1e-6, 1.0, rowsum)
    A = A / rowsum
    A[~np.isfinite(A)] = 0.0
    return A


# ------------------------- time & weather exogenous -------------------------

def add_exogenous_basic(
    times: pd.DatetimeIndex,
    E: int,
    weather_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Basic time-of-day/day-of-week (+ optional weather) features,
    repeated across E edges. Returns [T, E, F_exog_basic].
    """
    times = pd.DatetimeIndex(times)
    T = len(times)

    # time-of-day (sin/cos)
    sec_in_day = 24 * 3600
    tod_seconds = (
        times.hour.to_numpy() * 3600
        + times.minute.to_numpy() * 60
        + times.second.to_numpy()
    )
    tod_angle = 2 * np.pi * tod_seconds / sec_in_day
    tod_sin = np.sin(tod_angle).reshape(T, 1).astype(np.float32)
    tod_cos = np.cos(tod_angle).reshape(T, 1).astype(np.float32)

    # day-of-week (sin/cos)
    dow = times.dayofweek.to_numpy()
    dow_angle = 2 * np.pi * dow / 7.0
    dow_sin = np.sin(dow_angle).reshape(T, 1).astype(np.float32)
    dow_cos = np.cos(dow_angle).reshape(T, 1).astype(np.float32)

    feats = [tod_sin, tod_cos, dow_sin, dow_cos]

    # weather aligned to timeline (optional)
    if weather_df is not None:
        W = weather_df.reindex(times).fillna(method="ffill").fillna(method="bfill")
        feats.append(W.to_numpy(dtype=np.float32, copy=False))

    exog_t = np.concatenate(feats, axis=1)           # [T, F_basic]
    exog = np.repeat(exog_t[:, None, :], E, axis=1)  # [T, E, F_basic]
    return exog


# ------------------------- lag features -------------------------

def add_edge_lags(X: np.ndarray, lags_steps: Iterable[int]) -> np.ndarray:
    """
    Self (per-edge) lags from raw or filled counts.
    X: [T,E]  -> returns [T,E,L]
    """
    T, E = X.shape
    lags_steps = list(lags_steps)
    feats = []
    for L in lags_steps:
        lag = np.zeros((T, E), dtype=np.float32)
        if L < T:
            lag[L:] = X[:-L]
        feats.append(lag[..., None])
    return np.concatenate(feats, axis=2) if feats else np.zeros((T, E, 0), np.float32)


def neighbor_lag_features(X: np.ndarray, A_bin: np.ndarray, lags=(1, 2, 3)) -> np.ndarray:
    """
    Neighbor lag features using *given* X (raw or filled).
    A_bin: binary edge-edge adjacency (1: neighbors in line graph). Returns [T,E,len(lags)].
    """
    T, E = X.shape
    feats = []
    deg = np.maximum(1.0, A_bin.sum(1))  # degree per edge
    for L in lags:
        lag = np.zeros((T, E), dtype=np.float32)
        if L < T:
            # neighbor values at t-L influence edge at t
            lag[L:] = (X[:-L] @ A_bin.T) / deg
        feats.append(lag[..., None])
    return np.concatenate(feats, axis=2) if feats else np.zeros((T, E, 0), np.float32)


# ------------------------- causal graph imputation -------------------------

def causal_graph_impute(
    X: np.ndarray,
    M: np.ndarray,
    A_bin: np.ndarray,
    alpha_ema: float = 0.7,
    w_self: float = 0.7,
    w_nei: float = 0.3,
) -> np.ndarray:
    """
    Causal imputation for missing/no-sensor bins.
    Uses an EMA of each edge and the previous-step neighbor mean.
    All operations use information <= t-1.

    X: [T,E] counts (zeros ok)
    M: [T,E] mask (1 if observed, else 0)
    A_bin: [E,E] binary adjacency of line-graph
    Returns X_filled: [T,E]
    """
    T, E = X.shape
    Xf = np.zeros_like(X, dtype=np.float32)
    ema = np.zeros(E, dtype=np.float32)
    deg = np.maximum(1.0, A_bin.sum(1))

    for t in range(T):
        # neighbor signal from previous step
        if t == 0:
            nei_prev = np.zeros(E, dtype=np.float32)
        else:
            nei_prev = (Xf[t - 1] @ A_bin.T) / deg

        # candidate fill for missing values
        fill = w_self * ema + w_nei * nei_prev

        # choose observed vs fill
        x_t = np.where(M[t] > 0.5, X[t], fill)
        x_t = np.clip(x_t, 0.0, None)  # no negative counts

        Xf[t] = x_t

        # update EMA after fixing Xf[t]
        ema = alpha_ema * ema + (1.0 - alpha_ema) * Xf[t]

    return Xf


def causal_graph_impute_with_preds(
    X: np.ndarray,
    M: np.ndarray,
    A_bin: np.ndarray,
    Xhat_prev: Optional[np.ndarray] = None,
    alpha_ema: float = 0.7,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),  # (EMA, neighbor, pred)
) -> np.ndarray:
    """
    Like causal_graph_impute, but can blend in previous *predictions* causally.
    Xhat_prev is expected to be aligned [T,E] and only its t-1 row influences time t.
    """
    w_ema, w_nei, w_pred = weights
    T, E = X.shape
    Xf = np.zeros_like(X, dtype=np.float32)
    ema = np.zeros(E, dtype=np.float32)
    deg = np.maximum(1.0, A_bin.sum(1))

    for t in range(T):
        nei_prev = (Xf[t - 1] @ A_bin.T) / deg if t > 0 else np.zeros(E, np.float32)
        pred_prev = Xhat_prev[t - 1] if (Xhat_prev is not None and t > 0) else np.zeros(E, np.float32)

        fill = w_ema * ema + w_nei * nei_prev + w_pred * pred_prev
        x_t = np.where(M[t] > 0.5, X[t], fill)
        x_t = np.clip(x_t, 0.0, None)

        Xf[t] = x_t
        ema = alpha_ema * ema + (1.0 - alpha_ema) * Xf[t]

    return Xf


# ------------------------- master exogenous builder -------------------------

def add_exogenous(
    times: pd.DatetimeIndex,
    E: int,
    config: "Config",
    weather_df: Optional[pd.DataFrame] = None,
    *,
    # graph-aware extras (optional; if not provided, we fall back to basic features only)
    X: Optional[np.ndarray] = None,          # [T,E] counts (raw)
    M: Optional[np.ndarray] = None,          # [T,E] mask (1 if observed)
    A_bin: Optional[np.ndarray] = None,      # [E,E] binary adjacency
    use_graph_fill: bool = True,
    lags_self: Iterable[int] = (1, 2, 3, 6, 12),
    lags_neigh: Iterable[int] = (1, 2, 3),
    ema_alpha: float = 0.7,
    fill_weights: Tuple[float, float, float] = (0.7, 0.3, 0.0),  # (EMA, neighbor, pred)
    Xhat_prev: Optional[np.ndarray] = None,  # [T,E] optional previous predictions for causal fill
) -> np.ndarray:
    """
    Build exogenous features [T,E,F] combining:
      - basic time/day (and optional weather) features
      - self-lag features from *filled* series
      - neighbor-lag features from *filled* series

    If X/M/A_bin are None, returns only basic features (backward compatible).

    Args:
      times: timeline (DatetimeIndex of length T)
      E: number of directed edges
      config: unused here but kept for signature compatibility
      weather_df: optional weather aligned to 'times'
      X, M, A_bin: to enable graph-aware lags (recommended)
      use_graph_fill: if True, fill missing/no-sensor bins causally before computing lags
      lags_self: iterable of self-lag steps (in *time steps*, not hours)
      lags_neigh: iterable of neighbor-lag steps (in *time steps*)
      ema_alpha, fill_weights, Xhat_prev: control causal filling; set nonzero pred weight to use predictions

    Returns:
      exog: [T,E,F_exog]
    """
    times = pd.DatetimeIndex(times)
    T = len(times)

    # 1) Basic time/weather features
    exog_basic = add_exogenous_basic(times, E, weather_df)  # [T,E,F_basic]

    # 2) If no graph/time-series matrices provided, return basics only
    if X is None or M is None or A_bin is None:
        return exog_basic

    # 3) Causal fill to create a dense series for lags (works for no-sensor edges too)
    if use_graph_fill:
        if (Xhat_prev is not None) and (fill_weights[2] > 0.0):
            X_filled = causal_graph_impute_with_preds(
                X, M, A_bin, Xhat_prev=Xhat_prev,
                alpha_ema=ema_alpha, weights=fill_weights
            )
        else:
            # pure EMA + neighbor
            X_filled = causal_graph_impute(
                X, M, A_bin, alpha_ema=ema_alpha,
                w_self=fill_weights[0], w_nei=fill_weights[1]
            )
    else:
        # Just use raw X (zeros where missing). Still causal when we build lags.
        X_filled = np.array(X, dtype=np.float32, copy=True)

    # 4) Self- and neighbor-lag features from filled series
    self_l = add_edge_lags(X_filled, lags_steps=lags_self)           # [T,E,Ls]
    nei_l  = neighbor_lag_features(X_filled, A_bin, lags=lags_neigh) # [T,E,Ln]

    # 5) Concatenate all exogenous channels
    exog = np.concatenate([exog_basic, self_l, nei_l], axis=2)       # [T,E,F]
    return exog
