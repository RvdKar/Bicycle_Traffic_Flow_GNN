from __future__ import annotations
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple
# from scripts.preprocessing import NetworkConfig


def draw_network_flows(edge_values: Dict[Tuple[str,str], float], network_cfg: NetworkConfig, title: str=""):
    G = nx.DiGraph()
    G.add_nodes_from(network_cfg.nodes)
    for u,v in network_cfg.directed_edges:
        G.add_edge(u,v)
    plt.figure(figsize=(8,6))
    widths = []
    for (u,v) in G.edges():
        val = edge_values.get((u,v), 0.0)
        widths.append(1.0 + 0.06*max(0.0, val))
    nx.draw_networkx_nodes(G, network_cfg.pos, node_size=400, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, network_cfg.pos, font_size=10)
    nx.draw_networkx_edges(G, network_cfg.pos, width=widths, arrows=True, arrowstyle='-|>', arrowsize=12)
    plt.title(title); plt.axis('off'); plt.tight_layout(); plt.show()