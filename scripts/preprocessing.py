from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    dirs = []
    for u,v in undirected_edges:
        dirs.append((u,v))
        dirs.append((v,u))
    return dirs


def is_no_sensor_dir(e: Tuple[str,str], network_cfg: NetworkConfig) -> bool:
    NO_SENSOR_UNDIR = network_cfg.no_sensor_undir
    u,v = e
    return (u,v) in NO_SENSOR_UNDIR or (v,u) in NO_SENSOR_UNDIR


def build_line_graph_adjacency(directed_edges: List[Tuple[str,str]], allow_uturn=False) -> np.ndarray:
    """
    L(G) nodes = directed edges of original graph.
    Connect e1=(a->b) to e2=(c->d) if b==c and (optionally) d!=a to avoid U-turns.
    """
    E = len(directed_edges)
    A = np.zeros((E,E), dtype=np.float32)
    for i,(a,b) in enumerate(directed_edges):
        for j,(c,d) in enumerate(directed_edges):
            if b == c and (allow_uturn or d != a):
                A[i,j] = 1.0
    # add self-loops (common in GCN normalisation)
    np.fill_diagonal(A, 1.0)
    return A


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


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """ Ā = D^{-1/2} A D^{-1/2} """
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=(d>0))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def dedupe_within_bucket(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    # df has: timestamp, sensor_key, count_in, count_out, created_at (optional), end_time (optional)
    df = df.copy()
    df['bucket'] = df['timestamp'].dt.floor(bucket)

    sort_cols = ['sensor_key', 'bucket']
    # Prefer created_at if present, else end_time, else timestamp
    if 'created_at' in df.columns:
        sort_cols.append('created_at')
    elif 'end_time' in df.columns:
        sort_cols.append('end_time')
    else:
        sort_cols.append('timestamp')

    df = df.sort_values(sort_cols)
    # keep='last' -> keeps the most recent record per sensor_key per bucket
    df = df.drop_duplicates(subset=['sensor_key', 'bucket'], keep='last')

    df['timestamp'] = df['bucket']
    return df.drop(columns='bucket')


def parse_smartcamera_csv(paths: List[str]) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with:
      timestamp (naive datetime), sensor_key, count_in, count_out  (floats)
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # datetimes
        for col in ['start_time','end_time','created_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        # choose start_time as the bin anchor
        ts = df['start_time']
        if getattr(ts.dt, 'tz', None) is not None:
            ts = ts.dt.tz_convert(None)
        df['timestamp'] = ts

        df['sensor_key'] = df['sensor_id'].astype(str) + "|" + df['element_name'].astype(str)
        # numeric, NaN -> 0
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
    Make X[T,E] by routing count_in to (u->v) and count_out to (v->u)
    according to sensor_to_base_dir[sensor_key] = (u,v).
    Also builds a mask M[T,E] that is 1 only where targets are observed.
    """


    all_df = dedupe_within_bucket(all_df, config.time_granularity)
    df = all_df[all_df['sensor_key'].isin(network_cfg.sensor_to_edge.keys())].copy()
    if df.empty:
        raise ValueError("No sensor_key in CSV matched SENSOR_TO_EDGE mapping.")

    # Map base direction (u,v)
    uv = df['sensor_key'].map(network_cfg.sensor_to_edge)
    df['u'] = uv.apply(lambda t: t[0])
    df['v'] = uv.apply(lambda t: t[1])

    # Long form for IN: edge = (u->v)
    din = df[['timestamp','u','v','count_in']].rename(
        columns={'u':'src','v':'dst','count_in':'count'})
    # Long form for OUT: edge = (v->u)
    dout = df[['timestamp','u','v','count_out']].rename(
        columns={'u':'dst','v':'src','count_out':'count'})

    long = pd.concat([din, dout], ignore_index=True)
    # Pack tuple so we can pivot
    long['edge'] = list(zip(long['src'], long['dst']))

    # Sum within bins per directed edge
    piv = (long
           .pivot_table(index='timestamp', columns='edge', values='count', aggfunc='sum')
           .sort_index()
           .resample(config.time_granularity).sum()
           .fillna(0.0))

    # Ensure we have every directed edge as a column (zeros if absent)
    cols = [tuple(e) for e in network_cfg.directed_edges]
    for e in cols:
        if e not in piv.columns:
            piv[e] = 0.0
    piv = piv[cols]

    # Mask: 0 for edges that **never** have sensors; 1 elsewhere
    observed_mask = pd.DataFrame(1.0, index=piv.index, columns=piv.columns, dtype=float)
    for e in network_cfg.directed_edges:
        if is_no_sensor_dir(e, network_cfg):
            observed_mask[e] = 0.0

    # If you want to downweight sporadic missing bins for edges that *do* have sensors,
    # you can set mask=0 where both in and out were NaN before fill, but for counts
    # we usually keep zeros as valid 'no crossings' observations.

    times = piv.index
    X = piv.values.astype(np.float32)           # [T,E]
    M = observed_mask.values.astype(np.float32) # [T,E]
    return times, X, M


class WindowedEdgeDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, exog: Optional[np.ndarray], H_in: int, H_out: int):
        """
        X: [T, E] main feature (flow); exog: [T, E, F_exog] or None
        """
        assert X.ndim == 2
        self.X = X
        self.M = M
        self.exog = exog
        self.H_in = H_in
        self.H_out = H_out
        self.T, self.E = X.shape
        self.N = self.T - (H_in + H_out) + 1
        if self.N <= 0:
            raise ValueError("Not enough time steps for given H_in/H_out.")

    def __len__(self): return self.N

    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + self.H_in
        t2 = t1 + self.H_out
        x_hist = self.X[t0:t1, :]                         # [H_in, E]
        y_fut  = self.X[t1:t2, :]                         # [H_out, E]
        m_fut  = self.M[t1:t2, :]                         # [H_out, E]
        if self.exog is not None:
            ex_hist = self.exog[t0:t1, :, :]              # [H_in, E, F_exog]
            ex_fut  = self.exog[t1:t2, :, :]
            x = np.concatenate([x_hist[:,:,None], ex_hist], axis=2)  # [H_in, E, 1+F_exog]
            x_future_feats = ex_fut                                      # for decoder-free TCN we can pass future exog, or ignore
        else:
            x = x_hist[:,:,None]                          # [H_in, E, 1]
            x_future_feats = None
        return {
            "x": torch.from_numpy(x.astype(np.float32)),                 # [H_in, E, F]
            "y": torch.from_numpy(y_fut.astype(np.float32)),             # [H_out, E]
            "mask": torch.from_numpy(m_fut.astype(np.float32)),          # [H_out, E]
            "x_future_feats": (None if x_future_feats is None else torch.from_numpy(x_future_feats.astype(np.float32)))
        }