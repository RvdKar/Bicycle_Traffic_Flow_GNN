from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from typing import Dict, Tuple, TYPE_CHECKING
import src.preprocessing as preprocessing
if TYPE_CHECKING:
    from .preprocessing import NetworkConfig


def _robust_widths(values, base=1.0, scale=4.0):
    vals = np.array([max(0.0, v) for v in values], dtype=float)
    if len(vals) == 0: return []
    q = np.quantile(vals, 0.95) if np.any(vals>0) else 1.0
    return [base + scale * np.sqrt(v / (q + 1e-9)) for v in vals]

def draw_network_flows(edge_values: Dict[Tuple[str,str], float],
                       network_cfg: "NetworkConfig", title: str=""):
    G = nx.DiGraph()
    G.add_nodes_from(network_cfg.nodes)
    for u,v in network_cfg.directed_edges: G.add_edge(u,v)
    plt.figure(figsize=(8,6))
    vals = [edge_values.get((u,v), 0.0) for (u,v) in G.edges()]
    widths = _robust_widths(vals, base=0.8, scale=4.0)
    nx.draw_networkx_nodes(G, network_cfg.pos, node_size=420, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, network_cfg.pos, font_size=10)
    nx.draw_networkx_edges(G, network_cfg.pos, width=widths, arrows=True, arrowstyle='-|>', arrowsize=12)
    plt.title(title); plt.axis('off'); plt.tight_layout(); plt.show()


def draw_sensor_coverage_network(network_cfg,
                                 title: str | None = None,
                                 supervised_color: str = "#2878b5",
                                 unsupervised_color: str = "#b0b0b0",
                                 width: float = 2.8):
    """
    Undirected coverage map:
      - Edges with sensors: solid, supervised_color
      - Edges without sensors: dashed, unsupervised_color
    Uses network_cfg.physical_edges (undirected list) and network_cfg.no_sensor_undir (undirected set).
    """
    def undirected(e: Tuple[str,str]) -> Tuple[str,str]:
        u,v = e
        return (u,v) if u <= v else (v,u)

    phys_undir: Set[Tuple[str,str]] = {undirected(e) for e in network_cfg.physical_edges}
    no_sensor_undir: Set[Tuple[str,str]] = {undirected(e) for e in network_cfg.no_sensor_undir}
    supervised_undir: Set[Tuple[str,str]] = phys_undir - no_sensor_undir

    G = nx.Graph()
    G.add_nodes_from(network_cfg.nodes)
    G.add_edges_from(phys_undir)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, network_cfg.pos, node_size=420, node_color="white",
                           edgecolors="black", linewidths=1.0)

    nx.draw_networkx_edges(G, network_cfg.pos, edgelist=list(supervised_undir),
                           width=width, edge_color=supervised_color)
    nx.draw_networkx_edges(G, network_cfg.pos, edgelist=list(no_sensor_undir),
                           width=width, style="--", edge_color=unsupervised_color)

    nx.draw_networkx_labels(G, network_cfg.pos, font_size=10)

    if title:
        plt.title(title)
    plt.axis("off")

    handles = [
        Line2D([0],[0], color=supervised_color, lw=3.0, label="supervised"),
        Line2D([0],[0], color=unsupervised_color, lw=3.0, ls="--", label="un-supervised (no sensor)"),
    ]
    plt.legend(handles=handles, loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()

def compute_split_day_summary(times: pd.DatetimeIndex,
                              masks: dict,
                              *,
                              holiday_ranges=None,
                              exam_ranges=None) -> pd.DataFrame:
    """
    Return a table with one row per split (train/val/test):
      - days: number of unique days in that split
      - weekend/holiday/exam: percent of days in that split having that flag
    """
    times = pd.DatetimeIndex(times)
    day_of = times.normalize()
    days_all = day_of.unique().sort_values()

    # flags per timestamp, then "any in day" -> daily flags
    flags = preprocessing.calendar_flags(times, holiday_ranges=holiday_ranges, exam_ranges=exam_ranges)
    daily_flags = flags.groupby(day_of).max()  # one row per day

    def _to_bool(a):
        return a.to_numpy(dtype=bool) if hasattr(a, "to_numpy") else a.astype(bool)

    def summarize(mask_like) -> dict:
        mask = _to_bool(mask_like)
        sel_days = pd.Index(np.unique(day_of[mask]))
        if len(sel_days) == 0:
            return {"days": 0, "weekend_pct": 0.0, "holiday_pct": 0.0, "exam_pct": 0.0}
        df = daily_flags.reindex(sel_days).fillna(0.0)
        return {
            "days": int(len(sel_days)),
            "weekend_pct": 100.0 * float(df["is_weekend"].mean()),
            "holiday_pct": 100.0 * float(df["is_holiday"].mean()),
            "exam_pct":    100.0 * float(df["is_exam"].mean()),
        }

    rows = []
    for split in ("train", "val", "test"):
        rows.append({"split": split, **summarize(masks[split])})

    out = pd.DataFrame(rows)
    out.attrs["days_total"] = int(len(days_all))
    return out

def print_split_day_summary(times: pd.DatetimeIndex,
                            masks: dict,
                            *,
                            holiday_ranges=None,
                            exam_ranges=None) -> None:
    """
    Pretty-print like:
    Days total: 121
    train days= 85 | weekend=29.4% holiday=15.3% exam=18.8%
    ...
    """
    df = compute_split_day_summary(times, masks,
                                   holiday_ranges=holiday_ranges,
                                   exam_ranges=exam_ranges)
    print(f"Days total: {df.attrs.get('days_total', 'NA')}")
    for _, r in df.iterrows():
        print(f"{r['split']:5s} days={int(r['days']):3d} | "
              f"weekend={r['weekend_pct']:.1f}% "
              f"holiday={r['holiday_pct']:.1f}% "
              f"exam={r['exam_pct']:.1f}%")