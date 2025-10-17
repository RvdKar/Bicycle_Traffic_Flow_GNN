from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Iterable, Tuple
import random, numpy as np, pandas as pd, torch
if TYPE_CHECKING:
    from .preprocessing import Config

# ------------- utils -------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def steps_for_hours(granularity: str, hours: int) -> int:
    step = pd.Timedelta(granularity)
    return int(pd.Timedelta(f"{hours}h") / step)

def sanitize_and_row_normalize_A(A_hat_np: np.ndarray) -> np.ndarray:
    A = np.array(A_hat_np, dtype=np.float32, copy=True)
    A[~np.isfinite(A)] = 0.0
    np.fill_diagonal(A, A.diagonal() + 1.0)
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < 1e-6, 1.0, rowsum)
    A = A / rowsum
    A[~np.isfinite(A)] = 0.0
    return A

# ------------- basic exog -------------
def add_exogenous_basic(times: pd.DatetimeIndex,
                        E: int,
                        weather_df: Optional[pd.DataFrame] = None,
                        calendar_flags: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Build time-of-day, day-of-week sin/cos; optionally append weather columns and
    binary calendar flags (is_weekend, is_holiday, is_exam).
    Returns [T, E, F_basic].
    """
    times = pd.DatetimeIndex(times); T = len(times)
    sec_in_day = 24*3600
    s = (times.hour*3600 + times.minute*60 + times.second).to_numpy()
    theta = 2*np.pi*s/sec_in_day
    tod_sin = np.sin(theta).reshape(T,1).astype(np.float32)
    tod_cos = np.cos(theta).reshape(T,1).astype(np.float32)
    dow = times.dayofweek.to_numpy()
    phi = 2*np.pi*dow/7.0
    dow_sin = np.sin(phi).reshape(T,1).astype(np.float32)
    dow_cos = np.cos(phi).reshape(T,1).astype(np.float32)

    feats = [tod_sin, tod_cos, dow_sin, dow_cos]

    if weather_df is not None and not weather_df.empty:
        W = weather_df.reindex(times).ffill().bfill()
        feats.append(W.to_numpy(dtype=np.float32, copy=False))

    if calendar_flags is not None and not calendar_flags.empty:
        CF = calendar_flags.reindex(times).fillna(0.0)
        feats.append(CF.to_numpy(dtype=np.float32, copy=False))

    exog_t = np.concatenate(feats, axis=1) if len(feats) > 0 else np.zeros((T,0), np.float32)
    return np.repeat(exog_t[:, None, :], E, axis=1)

# ------------- lag features -------------
def add_edge_lags(X: np.ndarray, lags_steps: Iterable[int]) -> np.ndarray:
    T, E = X.shape; feats = []
    for L in list(lags_steps):
        lag = np.zeros((T,E), np.float32)
        if L < T: lag[L:] = X[:-L]
        feats.append(lag[...,None])
    return np.concatenate(feats, axis=2) if feats else np.zeros((T,E,0), np.float32)

def neighbor_lag_features(X: np.ndarray, A_bin: np.ndarray, lags=(1,2,3)) -> np.ndarray:
    T, E = X.shape; feats = []
    deg = np.maximum(1.0, A_bin.sum(1))
    for L in lags:
        lag = np.zeros((T,E), np.float32)
        if L < T: lag[L:] = (X[:-L] @ A_bin.T) / deg
        feats.append(lag[...,None])
    return np.concatenate(feats, axis=2) if feats else np.zeros((T,E,0), np.float32)

# ------------- causal imputation -------------
def causal_graph_impute(X: np.ndarray, M: np.ndarray, A_bin: np.ndarray,
                        alpha_ema: float = 0.7, w_self: float = 0.7, w_nei: float = 0.3) -> np.ndarray:
    T,E = X.shape; Xf = np.zeros_like(X, np.float32); ema = np.zeros(E, np.float32)
    deg = np.maximum(1.0, A_bin.sum(1))
    for t in range(T):
        nei_prev = (Xf[t-1] @ A_bin.T) / deg if t>0 else np.zeros(E, np.float32)
        fill = w_self*ema + w_nei*nei_prev
        x_t = np.where(M[t] > 0.5, X[t], fill).clip(0.0)
        Xf[t] = x_t
        ema = alpha_ema*ema + (1.0-alpha_ema)*Xf[t]
    return Xf

def causal_graph_impute_with_preds(X: np.ndarray, M: np.ndarray, A_bin: np.ndarray,
                                   Xhat_prev: Optional[np.ndarray] = None,
                                   alpha_ema: float = 0.7,
                                   weights: Tuple[float,float,float] = (0.5,0.3,0.2)) -> np.ndarray:
    w_ema, w_nei, w_pred = weights
    T,E = X.shape; Xf = np.zeros_like(X, np.float32); ema = np.zeros(E, np.float32)
    deg = np.maximum(1.0, A_bin.sum(1))
    for t in range(T):
        nei_prev  = (Xf[t-1] @ A_bin.T) / deg if t>0 else np.zeros(E, np.float32)
        pred_prev = Xhat_prev[t-1] if (Xhat_prev is not None and t>0) else np.zeros(E, np.float32)
        fill = w_ema*ema + w_nei*nei_prev + w_pred*pred_prev
        x_t = np.where(M[t] > 0.5, X[t], fill).clip(0.0)
        Xf[t] = x_t
        ema = alpha_ema*ema + (1.0-alpha_ema)*Xf[t]
    return Xf

# ------------- master exogenous -------------
def add_exogenous(times: pd.DatetimeIndex, E: int, config: "Config",
                  weather_df: Optional[pd.DataFrame] = None, *,
                  calendar_flags: Optional[pd.DataFrame] = None,
                  X: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, A_bin: Optional[np.ndarray] = None,
                  use_graph_fill: bool = True, lags_self: Iterable[int] = (1,2,3,6,12), lags_neigh: Iterable[int]=(1,2,3),
                  ema_alpha: float = 0.7, fill_weights: Tuple[float,float,float]=(0.7,0.3,0.0),
                  Xhat_prev: Optional[np.ndarray] = None) -> np.ndarray:
    times = pd.DatetimeIndex(times); T = len(times)
    exog_basic = add_exogenous_basic(times, E, weather_df, calendar_flags=calendar_flags)
    if X is None or M is None or A_bin is None:
        return exog_basic
    if use_graph_fill:
        if (Xhat_prev is not None) and (fill_weights[2] > 0.0):
            Xf = causal_graph_impute_with_preds(X, M, A_bin, Xhat_prev=Xhat_prev,
                                                alpha_ema=ema_alpha, weights=fill_weights)
        else:
            Xf = causal_graph_impute(X, M, A_bin, alpha_ema=ema_alpha,
                                     w_self=fill_weights[0], w_nei=fill_weights[1])
    else:
        Xf = np.array(X, dtype=np.float32, copy=True)
    self_l = add_edge_lags(Xf, lags_self)
    nei_l  = neighbor_lag_features(Xf, A_bin, lags_neigh)
    return np.concatenate([exog_basic, self_l, nei_l], axis=2)


def steps_for_hours(granularity: str, hours: int) -> int:
    step = pd.Timedelta(granularity)
    return int(pd.Timedelta(f"{hours}h") / step)
