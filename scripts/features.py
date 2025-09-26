from __future__ import annotations
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
# from preprocessing import Config


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def add_exogenous(times: pd.DatetimeIndex, E: int, config: Config, weather_df: Optional[pd.DataFrame]=None) -> np.ndarray:
    """
    Returns exogenous features repeated across edges: [T, E, F_exog]
    """
    # Ensure DatetimeIndex
    times = pd.DatetimeIndex(times)
    T = len(times)

    # --- time-of-day (seconds) ---
    sec_in_day = 24 * 3600
    tod_seconds = (
        times.hour.to_numpy() * 3600
        + times.minute.to_numpy() * 60
        + times.second.to_numpy()
    )

    tod_angle = 2 * np.pi * tod_seconds / sec_in_day
    tod_sin = np.sin(tod_angle).reshape(T, 1).astype(np.float32)
    tod_cos = np.cos(tod_angle).reshape(T, 1).astype(np.float32)

    # --- day-of-week (0=Mon) ---
    dow = times.dayofweek.to_numpy()
    dow_angle = 2 * np.pi * dow / 7.0
    dow_sin = np.sin(dow_angle).reshape(T, 1).astype(np.float32)
    dow_cos = np.cos(dow_angle).reshape(T, 1).astype(np.float32)

    feats = [tod_sin, tod_cos, dow_sin, dow_cos]

    # --- optional weather aligned to 'times' ---
    if weather_df is not None:
        W = (weather_df
             .reindex(times)
             .fillna(method='ffill')
             .fillna(method='bfill'))
        Wv = W.to_numpy(dtype=np.float32, copy=False)
        feats.append(Wv)

    exog_t = np.concatenate(feats, axis=1)            # [T, F_exog]
    exog = np.repeat(exog_t[:, None, :], E, axis=1)   # [T, E, F_exog]
    return exog


def add_edge_lags(X: np.ndarray, lags_steps=(12*24, 12*24*7)) -> np.ndarray:
    """
    Create edge-specific lag features from raw counts.
    X: [T, E] counts
    lags_steps: tuple of integer steps (for 5-min bins: 12*24=1 day, 12*24*7=1 week)
    returns: [T, E, L] where L=len(lags_steps)
    """
    T, E = X.shape
    feats = []
    for L in lags_steps:
        lag = np.zeros((T, E), dtype=np.float32)   # zeros for the first L rows (no past)
        if L < T:
            lag[L:] = X[:-L]
        feats.append(lag[..., None])                # [T,E,1]
    return np.concatenate(feats, axis=2)            # [T,E,L]


def steps_for_hours(granularity: str, hours: int) -> int:
    step = pd.Timedelta(granularity)          # e.g., '5min' or '15min'
    return int(pd.Timedelta(f'{hours}h') / step)


def sanitize_and_row_normalize_A(A_hat_np: np.ndarray) -> np.ndarray:
    A = np.array(A_hat_np, dtype=np.float32, copy=True)

    # Remove non-finite, then add self-loops (guarantees >=1 degree)
    A[~np.isfinite(A)] = 0.0
    np.fill_diagonal(A, A.diagonal() + 1.0)

    # Row-normalize with floor to avoid div-by-zero
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < 1e-6, 1.0, rowsum)
    A = A / rowsum

    # Final clean
    A[~np.isfinite(A)] = 0.0
    return A
