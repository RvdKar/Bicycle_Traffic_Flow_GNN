from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any, Optional, Iterable
import numpy as np
import pandas as pd
import torch
from zoneinfo import ZoneInfo
from torch.utils.data import Dataset
import xarray as xr
from . import features, training


# ----------------------- network config -----------------------

@dataclass
class NetworkConfig:
    nodes: List[str]
    pos: Dict[str, Tuple[float, float]]
    physical_edges: List[Tuple[str, str]]
    no_sensor_undir: Set[Tuple[str, str]]
    sensor_to_edge: Dict[str, Tuple[str, str]]
    directed_edges: List[Tuple[str, str]]
    edge_index: Dict[Tuple[str, str], int]
    index_edge: Dict[int, Tuple[str, str]]
    a_line: Any
    a_hat: Any

def undir_to_dir(undirected_edges: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    return [(u,v) for (u,v) in undirected_edges] + [(v,u) for (u,v) in undirected_edges]

def is_no_sensor_dir(e: Tuple[str,str], network_cfg: NetworkConfig) -> bool:
    u,v = e
    S = network_cfg.no_sensor_undir
    return (u,v) in S or (v,u) in S

def build_line_graph_adjacency(directed_edges: List[Tuple[str,str]], allow_uturn: bool=False) -> np.ndarray:
    """
    L(G) nodes = directed edges of original graph.
    Connect e1=(a->b) to e2=(c->d) if b==c and (optionally) d!=a to avoid U-turns.
    NOTE: self-loops are NOT added here; add them in the normalizer once.
    """
    E = len(directed_edges)
    A = np.zeros((E,E), dtype=np.float32)
    for i,(a,b) in enumerate(directed_edges):
        for j,(c,d) in enumerate(directed_edges):
            if b == c and (allow_uturn or d != a):
                A[i,j] = 1.0
    return A

# ----------------------- runtime config -----------------------

@dataclass
class Config:
    time_granularity: str = "15min"
    H_in: int = 12
    H_out: int = 12
    batch_size: int = 64
    lr: float = 1e-4
    max_epochs: int = 20
    patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    use_exog: bool = True
    gcn_type: str = "cheb"   # or "gcn"
    cheb_K: int = 3
    nblocks: int = 2
    hidden: int = 32
    dropout: float = 0.12
    lambda_lap: float = 1e-3
    weight_decay: float = 0.0

# ----------------------- adjacency utils -----------------------

def normalize_adjacency_symmetric(A: np.ndarray) -> np.ndarray:
    """ Ā = D^{-1/2} (A + I) D^{-1/2} with I added exactly once. """
    A = A.astype(np.float32, copy=True)
    A[np.isnan(A)] = 0.0
    E = A.shape[0]
    A = A + np.eye(E, dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=(d>0))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)

def row_normalize_with_self_loops(A: np.ndarray) -> np.ndarray:
    """ Add I once, then row-normalize. """
    A = A.astype(np.float32, copy=True)
    E = A.shape[0]
    A = A + np.eye(E, dtype=np.float32)
    rowsum = A.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum < 1e-6, 1.0, rowsum)
    return (A / rowsum).astype(np.float32)

# ----------------------- CSV parsing & matrices -----------------------

def dedupe_within_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    df = df.copy()
    df['bucket'] = df['timestamp'].dt.floor(bucket)
    sort_cols = ['sensor_key', 'bucket']
    if 'created_at' in df.columns: sort_cols.append('created_at')
    elif 'end_time' in df.columns: sort_cols.append('end_time')
    else: sort_cols.append('timestamp')
    df = df.sort_values(sort_cols).drop_duplicates(subset=['sensor_key','bucket'], keep='last')
    df['timestamp'] = df['bucket']
    return df.drop(columns='bucket')

def parse_smartcamera_csv(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        for col in ['start_time','end_time','created_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        ts = df['start_time']
        if getattr(ts.dt, 'tz', None) is not None:
            ts = ts.dt.tz_convert(None)
        df['timestamp'] = ts
        df['sensor_key'] = df['sensor_id'].astype(str) + "|" + df['element_name'].astype(str)
        df['count_in']  = pd.to_numeric(df.get('count_in', 0), errors='coerce').fillna(0.0)
        df['count_out'] = pd.to_numeric(df.get('count_out',0), errors='coerce').fillna(0.0)
        frames.append(df[['timestamp','sensor_key','count_in','count_out']])
    return pd.concat(frames, ignore_index=True)

def build_edge_time_matrix(
    all_df: pd.DataFrame,
    config: Config,
    network_cfg: NetworkConfig
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Make X[T,E] by routing count_in to (u->v) and count_out to (v->u).
    M[T,E] = 0 for edges that never have sensors (no-sensor directions).
    """
    all_df = dedupe_within_bucket(all_df, config.time_granularity)
    df = all_df[all_df['sensor_key'].isin(network_cfg.sensor_to_edge.keys())].copy()
    if df.empty:
        raise ValueError("No sensor_key in CSV matched sensor_to_edge.")
    uv = df['sensor_key'].map(network_cfg.sensor_to_edge)
    df['u'] = uv.apply(lambda t: t[0]); df['v'] = uv.apply(lambda t: t[1])
    din  = df[['timestamp','u','v','count_in' ]].rename(columns={'u':'src','v':'dst','count_in':'count'})
    dout = df[['timestamp','u','v','count_out']].rename(columns={'u':'dst','v':'src','count_out':'count'})
    long = pd.concat([din, dout], ignore_index=True)
    long['edge'] = list(zip(long['src'], long['dst']))
    piv = (long.pivot_table(index='timestamp', columns='edge', values='count', aggfunc='sum')
                .sort_index()
                .resample(config.time_granularity).sum()
                .fillna(0.0))
    cols = [tuple(e) for e in network_cfg.directed_edges]
    for e in cols:
        if e not in piv.columns: piv[e] = 0.0
    piv = piv[cols]
    mask = pd.DataFrame(1.0, index=piv.index, columns=piv.columns, dtype=float)
    for e in network_cfg.directed_edges:
        if is_no_sensor_dir(e, network_cfg): mask[e] = 0.0
    times = piv.index
    X = piv.values.astype(np.float32)           # [T,E]
    M = mask.values.astype(np.float32)          # [T,E]
    return times, X, M

# ----------------------- dataset & splitting -----------------------

class WindowedEdgeDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, exog: Optional[np.ndarray], H_in: int, H_out: int,
                 valid_start_indices: Optional[Iterable[int]]=None):
        """
        X: [T, E]; exog: [T, E, F_exog] or None.
        If valid_start_indices is given, windows start only at these indices (gap-aware).
        """
        assert X.ndim == 2
        self.X, self.M, self.exog = X, M, exog
        self.H_in, self.H_out = H_in, H_out
        self.T, self.E = X.shape
        self.N = self.T - (H_in + H_out) + 1
        if self.N <= 0:
            raise ValueError("Not enough time steps for given H_in/H_out.")
        if valid_start_indices is None:
            self.starts = np.arange(self.N, dtype=np.int64)
        else:
            self.starts = np.array([s for s in valid_start_indices if 0 <= s < self.N], dtype=np.int64)

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        t0 = int(self.starts[idx]); t1 = t0 + self.H_in; t2 = t1 + self.H_out
        x_hist = self.X[t0:t1, :]                         # [H_in, E]
        y_fut  = self.X[t1:t2, :]                         # [H_out, E]
        m_fut  = self.M[t1:t2, :]                         # [H_out, E]
        if self.exog is not None:
            ex_hist = self.exog[t0:t1, :, :]              # [H_in, E, F_exog]
            ex_fut  = self.exog[t1:t2, :, :]
            x = np.concatenate([x_hist[:,:,None], ex_hist], axis=2)  # [H_in, E, 1+F_exog]
            x_future_feats = ex_fut
        else:
            x = x_hist[:,:,None]
            x_future_feats = None
        out = {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y_fut.astype(np.float32)),
            "mask": torch.from_numpy(m_fut.astype(np.float32)),
        }
        if x_future_feats is not None:
            out["x_future_feats"] = torch.from_numpy(x_future_feats.astype(np.float32))
        else:
            out["x_future_feats"] = None
        return out

def contiguous_ranges(index: pd.DatetimeIndex) -> List[Tuple[int,int]]:
    """
    Return [ (start_idx, end_idx_exclusive), ... ] of contiguous segments (no jumps > 1*granularity).
    """
    if len(index) == 0: return []
    dt = pd.Timedelta(index.freq) if index.freq is not None else pd.Timedelta(index[1]-index[0])
    gaps = np.where((index[1:] - index[:-1]) > dt)[0]
    starts = np.r_[0, gaps+1]; ends = np.r_[gaps+1, len(index)]
    return list(zip(starts.tolist(), ends.tolist()))

def valid_window_starts_from_ranges(ranges: List[Tuple[int,int]], H_in: int, H_out: int) -> List[int]:
    starts = []
    L = H_in + H_out
    for s,e in ranges:
        seg_len = e - s
        if seg_len >= L:
            starts.extend(list(range(s, e - L + 1)))
    return starts

# ----------------------- stratified split -----------------------

def day_category_index(times: pd.DatetimeIndex,
                       exam_ranges: Optional[List[Tuple[str, str]]] = None,
                       holiday_ranges: Optional[List[Tuple[str, str]]] = None) -> pd.Series:
    """
    Label each timestamp with a day 'category' using precedence:
      exam > holiday > weekend > weekday.
    Returns a pd.Series aligned to `times` with values in {"exam","holiday","weekend","weekday"}.
    """
    times = pd.DatetimeIndex(times)
    flags = calendar_flags(times, holiday_ranges=holiday_ranges, exam_ranges=exam_ranges)

    cat = np.full(len(times), "weekday", dtype=object)
    cat[np.where(times.dayofweek >= 5)] = "weekend"
    cat[np.where(flags["is_holiday"].values > 0.5)] = "holiday"
    cat[np.where(flags["is_exam"].values    > 0.5)] = "exam"

    return pd.Series(cat, index=times)


def _alloc_lrm(n: int, ratios: Tuple[float,float,float]) -> Tuple[int,int,int]:
    """Largest Remainder Method (Hamilton): floor then distribute remainders."""
    r = np.array(ratios, dtype=float)
    raw = r * n
    base = np.floor(raw).astype(int)
    give = int(n - base.sum())
    if give > 0:
        rem = raw - base
        order = np.argsort(-rem)  # largest remainders first
        for i in order[:give]:
            base[i] += 1
    return tuple(int(x) for x in base)

# def stratified_time_split_balanced(
#     times: pd.DatetimeIndex,
#     *,
#     train_ratio=0.70,
#     val_ratio=0.15,
#     test_ratio=0.15,
#     random_state: int = 42,
#     holiday_ranges=None,
#     exam_ranges=None,
# ) -> Dict[str, np.ndarray]:
#     """
#     Balance by COMPOSITE daily flags (is_weekend, is_holiday, is_exam).
#     For each unique daily signature (up to 8 combos), allocate days to splits
#     with the largest-remainder method, then build boolean masks on the full timeline.
#     """
#     rng = np.random.RandomState(random_state)
#     times = pd.DatetimeIndex(times)
#     day_of = times.normalize()

#     # daily flags (independent, no precedence)
#     flags = calendar_flags(times, holiday_ranges=holiday_ranges, exam_ranges=exam_ranges)
#     daily = flags.groupby(day_of).max()  # one row per day

#     # encode signature per day as 3-bit code: weekend<<0 | holiday<<1 | exam<<2
#     sig = (daily["is_weekend"].astype(int)
#            + 2 * daily["is_holiday"].astype(int)
#            + 4 * daily["is_exam"].astype(int))

#     # collect days by signature
#     days_by_sig: Dict[int, pd.DatetimeIndex] = {}
#     for code in np.sort(sig.unique()):
#         days_by_sig[int(code)] = sig.index[sig == code]

#     ratios = (train_ratio, val_ratio, test_ratio)
#     sel = {"train": set(), "val": set(), "test": set()}

#     def _alloc_lrm(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
#         raw = np.array(ratios, dtype=float) * n
#         base = np.floor(raw).astype(int)
#         give = int(n - base.sum())
#         if give > 0:
#             rem = raw - base
#             for i in np.argsort(-rem)[:give]:
#                 base[i] += 1
#         return int(base[0]), int(base[1]), int(base[2])

#     for code, days in days_by_sig.items():
#         n = len(days)
#         if n == 0:
#             continue
#         perm = days[rng.permutation(n)]
#         n_train, n_val, n_test = _alloc_lrm(n, ratios)
#         sel["train"].update(perm[:n_train])
#         sel["val"].update(perm[n_train:n_train + n_val])
#         sel["test"].update(perm[n_train + n_val:n_train + n_val + n_test])

#     def mask_for(split: str) -> np.ndarray:
#         if not sel[split]:
#             return np.zeros(len(times), dtype=bool)
#         sel_idx = pd.DatetimeIndex(list(sel[split]))
#         return np.isin(day_of.values, sel_idx.values)  # ndarray[bool]

#     return {
#         "train": mask_for("train"),
#         "val":   mask_for("val"),
#         "test":  mask_for("test"),
#     }

def stratified_time_split_balanced(
    times: pd.DatetimeIndex,
    *,
    ratios: Tuple[float,float,float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    flags_df: Optional[pd.DataFrame] = None,   # optional: provide precomputed flags with columns ["is_weekend","is_holiday","is_exam"]
    holiday_ranges: Optional[List[Tuple[str,str]]] = None,
    exam_ranges: Optional[List[Tuple[str,str]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Old-notebook style stratification:
      - label days by combined flags (W/H/E powerset)
      - Hamilton rounding inside each label bucket
      - global rebalance to hit exact totals (ratios * #days)
    Returns boolean masks over the FULL timeline (np.ndarray[bool] of len(times)).
    """
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    t = pd.DatetimeIndex(times)
    days = t.normalize()
    udays = days.unique().sort_values()

    # flags per timestamp
    if flags_df is None:
        # use your existing helper to build flags from ranges
        flags_df = calendar_flags(t, holiday_ranges=holiday_ranges, exam_ranges=exam_ranges)
    flags_df = flags_df.reindex(t).fillna(0.0)

    # per-day flags
    def day_flag(col):
        if col not in flags_df.columns:
            return pd.Series(0, index=udays, dtype=int)
        return (flags_df[col]
                .groupby(days).max()
                .reindex(udays).fillna(0).astype(int))

    W = day_flag("is_weekend")
    H = day_flag("is_holiday")
    E = day_flag("is_exam")

    # combined labels per day (powerset)
    labels = []
    for d in udays:
        parts = []
        if W.loc[d]: parts.append("W")
        if H.loc[d]: parts.append("H")
        if E.loc[d]: parts.append("E")
        labels.append("+".join(parts) if parts else "None")
    label = pd.Series(labels, index=udays, name="label")

    # Hamilton rounding inside each label bucket
    def _quota_counts(n, ratios_):
        raw = np.array(ratios_, float) * n
        base = np.floor(raw).astype(int)
        need = int(n - base.sum())
        if need > 0:
            rem = raw - base
            for j in np.argsort(rem)[::-1][:need]:
                base[j] += 1
        return base  # [train, val, test]

    train_days, val_days, test_days = [], [], []
    for L in label.unique():
        pool = udays[label == L].to_numpy()
        n = len(pool)
        if n == 0:
            continue
        q_tr, q_va, q_te = _quota_counts(n, ratios)
        perm = rng.permutation(pool)
        train_days.extend(perm[:q_tr])
        val_days.extend(perm[q_tr:q_tr+q_va])
        test_days.extend(perm[q_tr+q_va:q_tr+q_va+q_te])

    # disjoint sets
    train_days = set(pd.DatetimeIndex(train_days))
    val_days   = set(pd.DatetimeIndex(val_days)) - train_days
    test_days  = set(pd.DatetimeIndex(test_days)) - train_days - val_days
    sets = [train_days, val_days, test_days]

    # global targets (exact totals) via Hamilton rounding
    N = len(udays)
    raw_t = np.array(ratios, float) * N
    tgt   = np.floor(raw_t).astype(int)
    need  = int(N - tgt.sum())
    if need > 0:
        rem = raw_t - tgt
        for j in np.argsort(rem)[::-1][:need]:
            tgt[j] += 1
    target = tgt  # [train, val, test]

    # rebalance totals (keep sets disjoint)
    def _rebalance_sets(sets_, target_, rng_):
        def sizes(): return np.array([len(s) for s in sets_], dtype=int)
        cur = sizes()
        guard = 200000
        all_days = set().union(*sets_)
        while guard > 0 and not np.all(cur == target_):
            surplus = int(np.argmax(cur - target_))
            deficit = int(np.argmin(cur - target_))
            if cur[surplus] <= target_[surplus] or cur[deficit] >= target_[deficit]:
                break
            cand = list(sets_[surplus])
            if not cand:
                break
            d = cand[int(rng_.integers(len(cand)))]
            sets_[surplus].remove(d)
            sets_[deficit].add(d)
            cur = sizes()
            guard -= 1
        assert len(set().union(*sets_)) <= len(all_days)
        return sets_

    sets = _rebalance_sets(sets, target, rng)
    train_days, val_days, test_days = sets

    # masks over full timeline
    idx_train = np.isin(days.values, pd.DatetimeIndex(sorted(train_days)).values)
    idx_val   = np.isin(days.values, pd.DatetimeIndex(sorted(val_days)).values)
    idx_test  = np.isin(days.values, pd.DatetimeIndex(sorted(test_days)).values)

    return {"train": idx_train, "val": idx_val, "test": idx_test}

def to_ams_naive(ts: pd.Series) -> pd.Series:
    """
    Convert a datetime Series to Europe/Amsterdam and drop tzinfo
    so downstream pandas ops (weekday, resample) use local time.
    """
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
    return ts.dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)

# def load_davis_weather(nc_paths: List[str], time_granularity: str = "15min") -> pd.DataFrame:
#     """
#     Load Davis NetCDF files, convert time to Europe/Amsterdam (naive),
#     keep numeric columns, resample to `time_granularity`.
#     """
#     frames = []
#     for p in nc_paths:
#         ds = xr.open_dataset(p)
#         t = pd.to_datetime(ds["time"].values)
#         t = (pd.DatetimeIndex(t)
#                 .tz_localize("UTC")
#                 .tz_convert("Europe/Amsterdam")
#                 .tz_localize(None))
#         df = ds.to_dataframe().reset_index()
#         df["time"] = t
#         df = df.set_index("time").sort_index()
#         num = df.select_dtypes(include="number")
#         if not num.empty:
#             frames.append(num)
#     if not frames:
#         return pd.DataFrame(index=pd.DatetimeIndex([]))
#     W = pd.concat(frames).sort_index()
#     Wg = W.resample(time_granularity).mean()
#     # Optional normalisation of common names; harmless if absent
#     Wg = Wg.rename(columns={
#         "temp_out": "temp_C", "temp": "temp_C", "Temperature": "temp_C",
#         "rain": "rain_mm", "precip": "rain_mm", "rain_rate": "rain_mm_h",
#         "wind_speed": "wind_mps", "wind": "wind_mps", "wind_avg": "wind_mps",
#     })
#     return Wg

def _pick_var(ds, candidates):
    names = {n.lower(): n for n in ds.data_vars}
    for cand in candidates:
        for lname, real in names.items():
            if cand in lname:
                return real
    return None

def load_davis_weather(nc_paths: List[str], time_granularity: str = "5min") -> pd.DataFrame:
    """
    Read one or more Davis NetCDF files and return a tz-NAIVE Europe/Amsterdam dataframe,
    resampled to `time_granularity`, with THESE canonical columns (float32):

        temperature, wind_mps, wind_gust_speed, rain_mm, rain_mm_h

    - temperature, wind_mps:     interval means
    - wind_gust_speed:           interval max
    - rain_mm:                   mm per interval (from cumulative if available; else from rate)
    - rain_mm_h:                 mean rain rate in mm/h over the interval

    This preserves the robust variable-picking from your old loader and aligns to your
    new feature names & index conventions used elsewhere in the notebook.
    """
    AMS = ZoneInfo("Europe/Amsterdam")
    dfs = []

    for p in nc_paths:
        ds = xr.open_dataset(p)  # single-file; no dask needed

        # Pick available variables in THIS file
        var_temp      = _pick_var(ds, ['air_temp','airtemperature','temperature','temp_out','temp'])
        var_wind      = _pick_var(ds, ['wind_speed','windavg','wind','windspeed'])
        var_gust      = _pick_var(ds, ['wind_gust','windgust','gust'])
        var_rain_cum  = _pick_var(ds, ['rain_cum','rain_total','precip_accum','precipitation_accum'])
        var_rain_rate = _pick_var(ds, ['rain_rate','precip_rate','precipitation_rate'])

        keep = [v for v in [var_temp, var_wind, var_gust, var_rain_cum, var_rain_rate] if v]
        if not keep:
            continue

        df = ds[keep].to_dataframe().reset_index()

        # Normalize/convert time to AMS then make it NAIVE (matches the rest of your pipeline)
        time_col = 'time' if 'time' in df.columns else next(
            (c for c in df.columns if c.lower() in ('datetime','date','timestamp')), None
        )
        if time_col is None:
            raise RuntimeError(f"No time coordinate found in {p}")
        t = pd.to_datetime(df[time_col])
        if t.dt.tz is None:
            t = t.dt.tz_localize('UTC').dt.tz_convert(AMS)
        else:
            t = t.dt.tz_convert(AMS)
        t = t.dt.tz_localize(None)  # tz-naive AMS

        df = df.set_index(t).sort_index()
        df.index.name = "time"

        # Resample with appropriate aggregations
        agg = {}
        if var_temp:      agg[var_temp]      = 'mean'
        if var_wind:      agg[var_wind]      = 'mean'
        if var_gust:      agg[var_gust]      = 'max'
        if var_rain_rate: agg[var_rain_rate] = 'mean'
        if var_rain_cum:  agg[var_rain_cum]  = 'last'
        rs = df.resample(time_granularity).agg(agg)

        # Map to canonical names used in the NEW notebook
        out = pd.DataFrame(index=rs.index)
        if var_temp:      out['temperature']      = rs[var_temp]
        if var_wind:      out['wind_mps']         = rs[var_wind]
        if var_gust:      out['wind_gust_speed']  = rs[var_gust]
        if var_rain_rate: out['rain_mm_h']        = rs[var_rain_rate]
        if var_rain_cum:  out['rain_cum']         = rs[var_rain_cum]  # temp for diff

        dfs.append(out)

    if not dfs:
        # return empty with expected columns so downstream code is safe
        idx = pd.DatetimeIndex([], name="time")
        return pd.DataFrame(index=idx, columns=[
            "temperature","wind_mps","wind_gust_speed","rain_mm","rain_mm_h"
        ]).astype('float32')

    # Concatenate months and sort
    W = pd.concat(dfs, axis=0).sort_index()

    # Derive rain per interval (prefer cumulative if present)
    if 'rain_cum' in W.columns:
        W['rain_mm'] = W['rain_cum'].diff().clip(lower=0)
    elif 'rain_mm_h' in W.columns:
        minutes = pd.to_timedelta(time_granularity).components.minutes or 1
        W['rain_mm'] = W['rain_mm_h'] * (minutes / 60.0)
    else:
        W['rain_mm'] = np.nan

    # Ensure all expected columns exist
    for col in ["temperature","wind_mps","wind_gust_speed","rain_mm_h"]:
        if col not in W.columns:
            W[col] = np.nan

    # Final selection & dtype; drop helper
    cols = ["temperature","wind_mps","wind_gust_speed","rain_mm","rain_mm_h"]
    W = W[cols].astype('float32')
    if 'rain_cum' in W.columns:
        W = W.drop(columns=['rain_cum'], errors='ignore')
    return W


def align_weather_to_times(weather_df: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame | None:
    """
    Align weather to model timeline:
    - ensure DatetimeIndex, sorted, unique
    - reindex to times
    - forward/backward fill
    """
    if weather_df is None or weather_df.empty:
        return None

    w = weather_df.copy()
    # ensure datetime index and sorted
    if not isinstance(w.index, pd.DatetimeIndex):
        w.index = pd.to_datetime(w.index)
    w = w.sort_index()

    # collapse duplicates at identical timestamps (take last; or use mean())
    if w.index.has_duplicates:
        # w = w.groupby(level=0).mean()            # average if you prefer
        w = w[~w.index.duplicated(keep="last")]     # take last observation

    # reindex to target timeline and fill small gaps
    target = pd.DatetimeIndex(times)
    w = w.reindex(target)
    w = w.ffill().bfill()
    return w


def build_data_and_loaders(
    csv_paths: List[str],
    nc_paths: List[str],
    cfg: "Config",
    network_cfg,
    exam_ranges=None,
    holiday_ranges=None
) -> Dict[str, object]:
    """
    1) Load camera CSVs, convert to Europe/Amsterdam (naive)
    2) Build X, M
    3) Load Davis weather, resample, align to times
    4) Stratified split by day-category (weekday/weekend/exam)
    5) Calendar-aligned lags: 24h/7d self, 24h neighbour
    6) Build exogenous
    7) Build GAP-AWARE loaders
    """
    # --- 1. Camera CSVs ---
    raw_df = parse_smartcamera_csv(csv_paths)
    raw_df["timestamp"] = to_ams_naive(raw_df["timestamp"])

    # --- 2. Build edge–time matrices ---
    times, X, M = build_edge_time_matrix(raw_df, cfg, network_cfg)
    times = pd.DatetimeIndex(times)
    T, E = X.shape

    # --- 3. Weather ---
    weather_df = load_davis_weather(nc_paths, cfg.time_granularity) if nc_paths else None
    weather_df = align_weather_to_times(weather_df, times) if weather_df is not None else None

    # --- 4. Stratified splits ---
    cats = day_category_index(times,
                          exam_ranges=exam_ranges,
                          holiday_ranges=holiday_ranges)
    
    flags_df = calendar_flags(times,
                          holiday_ranges=holiday_ranges,
                          exam_ranges=exam_ranges)

    masks = stratified_time_split_balanced(
        times,
        ratios=(0.70, 0.15, 0.15),  # e.g., (0.70, 0.15, 0.15)
        seed=cfg.seed,
        flags_df=flags_df,                 # use the flags we already computed
        holiday_ranges=holiday_ranges,
        exam_ranges=exam_ranges,
    )

    # --- 5. Lags (calendar aligned) ---
    L_24h = features.steps_for_hours(cfg.time_granularity, 24)
    L_7d  = features.steps_for_hours(cfg.time_granularity, 24 * 7)

    # --- 6. Exogenous features ---
    A_line_bin = (network_cfg.a_line > 0).astype(np.float32)
    exog = features.add_exogenous(
        times=times,
        E=E,
        config=cfg,
        weather_df=weather_df,
        calendar_flags=flags_df,
        X=X,
        M=M,
        A_bin=A_line_bin,
        use_graph_fill=True,
        lags_self=(1,2,3, L_24h, L_7d),
        lags_neigh=(1,2),
        ema_alpha=0.7,
        fill_weights=(0.7, 0.3, 0.0),
        Xhat_prev=None,
    )

    # --- 7. Gap-aware loaders ---
    loaders = training.make_loaders_gapaware(times, X, M, exog, cfg, masks, shuffle_train=True)

    # --- 8. Return bundle ---
    return {
        "times": times,
        "X": X,
        "M": M,
        "T": T,
        "E": E,
        "weather_df": weather_df,
        "exog": exog,
        "cats": cats,
        "flags": flags_df,
        "masks": masks,
        "loaders": loaders,
        "lags": {"L_24h": L_24h, "L_7d": L_7d},
    }

def _ranges_to_mask(times: pd.DatetimeIndex,
                    ranges: Optional[List[Tuple[str, str]]]) -> pd.Series:
    """
    Build a boolean Series (index=times) that is True when the timestamp falls
    within ANY of the inclusive [start, end] string ranges (YYYY-MM-DD or ISO).
    """
    mask = pd.Series(False, index=times)
    if not ranges:
        return mask
    for s, e in ranges:
        s = pd.Timestamp(s)
        e = pd.Timestamp(e) + pd.Timedelta(days=0)  # inclusive end; timestamps are precise to minutes
        mask |= (times >= s) & (times <= e)
    return mask

def calendar_flags(times: pd.DatetimeIndex,
                   holiday_ranges: Optional[List[Tuple[str, str]]] = None,
                   exam_ranges: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      - is_weekend
      - is_holiday
      - is_exam
    Precedence is not applied here; these are independent flags.
    """
    times = pd.DatetimeIndex(times)
    is_weekend = (times.dayofweek >= 5)
    is_holiday = _ranges_to_mask(times, holiday_ranges).to_numpy()
    is_exam    = _ranges_to_mask(times, exam_ranges).to_numpy()
    return pd.DataFrame({
        "is_weekend": is_weekend.astype(np.float32),
        "is_holiday": is_holiday.astype(np.float32),
        "is_exam":    is_exam.astype(np.float32),
    }, index=times)