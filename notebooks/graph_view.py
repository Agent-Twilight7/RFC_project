"""
Step Graph View: Visualize Bayesian Network graphs for stages P1-P6.

Reads BN JSON files created by step11.py from data/processed/step11
and saves graph images to reports/figures/bn_graphs.
"""

import json
import os

import matplotlib.pyplot as plt
import networkx as nx

STAGES = ["P1", "P2", "P3", "P4", "P5", "P6"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BN_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "step11")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "bn_graphs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_bn_stage(stage):
    file_path = os.path.join(BN_DIR, f"BN_{stage}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_graph(bn_data):
    graph = nx.DiGraph()
    nodes = bn_data.get("nodes", [])
    edges = bn_data.get("edges", [])

    graph.add_nodes_from(nodes)
    graph.add_edges_from((edge["from"], edge["to"]) for edge in edges)
    return graph


def draw_and_save_stage(stage, graph):
    plt.figure(figsize=(10, 8))

    if graph.number_of_nodes() == 0:
        plt.text(0.5, 0.5, f"No nodes for {stage}", ha="center", va="center")
        plt.axis("off")
    else:
        layout = nx.spring_layout(graph, seed=42)
        nx.draw_networkx_nodes(graph, layout, node_size=900, alpha=0.95)
        nx.draw_networkx_labels(graph, layout, font_size=8)
        nx.draw_networkx_edges(
            graph,
            layout,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            width=1.3,
            alpha=0.8,
            connectionstyle="arc3,rad=0.08",
        )
        plt.title(f"Bayesian Network Graph - {stage}")
        plt.axis("off")

    out_path = os.path.join(OUT_DIR, f"BN_{stage}_graph.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[INFO] Saved graph for {stage}: {out_path}")


def draw_overview(stage_graphs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for index, stage in enumerate(STAGES):
        axis = axes[index]
        graph = stage_graphs.get(stage)

        if graph is None:
            axis.text(0.5, 0.5, f"Missing BN_{stage}.json", ha="center", va="center")
            axis.set_title(stage)
            axis.axis("off")
            continue

        if graph.number_of_nodes() == 0:
            axis.text(0.5, 0.5, "No nodes", ha="center", va="center")
            axis.set_title(stage)
            axis.axis("off")
            continue

        layout = nx.spring_layout(graph, seed=42)
        nx.draw_networkx_nodes(graph, layout, node_size=280, ax=axis, alpha=0.95)
        nx.draw_networkx_labels(graph, layout, font_size=6, ax=axis)
        nx.draw_networkx_edges(
            graph,
            layout,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            width=1.0,
            alpha=0.8,
            ax=axis,
            connectionstyle="arc3,rad=0.08",
        )
        axis.set_title(stage)
        axis.axis("off")

    fig.suptitle("Bayesian Network Graphs (P1-P6)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "BN_all_stages_overview.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] Saved combined overview: {out_path}")


def main():
    print(f"[INFO] Loading BN JSON files from: {BN_DIR}")
    stage_graphs = {}

    for stage in STAGES:
        bn_data = load_bn_stage(stage)
        if bn_data is None:
            print(f"[WARNING] Missing BN file for {stage}. Expected: BN_{stage}.json")
            stage_graphs[stage] = None
            continue

        graph = build_graph(bn_data)
        stage_graphs[stage] = graph
        print(
            f"[INFO] {stage}: nodes={graph.number_of_nodes()}, "
            f"edges={graph.number_of_edges()}"
        )
        draw_and_save_stage(stage, graph)

    draw_overview(stage_graphs)
    print(f"[DONE] Graph images saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
