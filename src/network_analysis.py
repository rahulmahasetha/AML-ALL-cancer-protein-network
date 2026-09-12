"""
Steps 7-8: Network Centrality Analysis & Combined Protein Ranking.

Computes degree, betweenness, and closeness centrality for all nodes
in the PPI network, using a positive distance transformation for weighted
metrics. Produces TWO rankings:
  1. All-network ranking (every node)
  2. AML-seed-only ranking (final candidate list)
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CENTRALITY_WEIGHTS,
    DISTANCE_EPSILON,
    TOP_N_PROTEINS,
    RESULTS_DIR,
)


def _add_distance_weights(G: nx.Graph):
    """
    Convert STRING combined_score to a positive distance:
        distance = 1 / (score/1000 + epsilon)

    Higher confidence → smaller distance.
    All values are strictly positive.
    """
    for u, v, data in G.edges(data=True):
        score = data.get("combined_score", 0)
        data["distance"] = 1.0 / (score / 1000.0 + DISTANCE_EPSILON)


def compute_centrality_and_rank(G: nx.Graph):
    """
    Compute centrality metrics and generate rankings.

    Parameters
    ----------
    G : nx.Graph
        PPI graph with node_type attribute ('aml_seed' or 'neighbor')
        and combined_score edge attribute.

    Returns
    -------
    centrality_all : pd.DataFrame
        Centrality table for ALL nodes.
    ranking_seeds : pd.DataFrame
        Ranked table for AML seed proteins only (the final candidates).
    """
    print("\n" + "=" * 60)
    print("STEP 7: NETWORK CENTRALITY ANALYSIS")
    print("=" * 60)

    if G.number_of_nodes() == 0:
        print("  ⚠ Network is empty – skipping centrality.")
        empty = pd.DataFrame()
        return empty, empty

    # --- Add distance weights ---
    _add_distance_weights(G)
    print(f"  Distance transformation: distance = 1 / (score/1000 + {DISTANCE_EPSILON})")

    # --- Degree ---
    print("  Computing degree centrality...")
    degree_cent = nx.degree_centrality(G)

    # --- Betweenness (weighted by distance) ---
    print("  Computing betweenness centrality (weighted)...")
    betweenness_cent = nx.betweenness_centrality(G, weight="distance", normalized=True)

    # --- Closeness (weighted by distance) ---
    print("  Computing closeness centrality (weighted)...")
    closeness_cent = nx.closeness_centrality(G, distance="distance")

    # --- Build results table ---
    records = []
    for node in G.nodes():
        records.append({
            "protein": node,
            "protein_name": G.nodes[node].get("protein_name", ""),
            "node_type": G.nodes[node].get("node_type", "unknown"),
            "degree": G.degree(node),
            "degree_centrality": degree_cent.get(node, 0.0),
            "betweenness": betweenness_cent.get(node, 0.0),
            "closeness": closeness_cent.get(node, 0.0),
        })

    centrality_all = pd.DataFrame(records)

    # --- Normalize to [0, 1] using min-max ---
    for col in ["degree_centrality", "betweenness", "closeness"]:
        col_min = centrality_all[col].min()
        col_max = centrality_all[col].max()
        norm_col = f"norm_{col}"
        if col_max > col_min:
            centrality_all[norm_col] = (
                (centrality_all[col] - col_min) / (col_max - col_min)
            )
        else:
            centrality_all[norm_col] = 0.0

    # --- Combined score ---
    w = CENTRALITY_WEIGHTS
    centrality_all["combined_score"] = (
        w["degree"] * centrality_all["norm_degree_centrality"]
        + w["betweenness"] * centrality_all["norm_betweenness"]
        + w["closeness"] * centrality_all["norm_closeness"]
    )

    # --- Step 8: Rankings ---
    print("\n" + "=" * 60)
    print("STEP 8: COMBINED PROTEIN RANKING")
    print("=" * 60)
    print(f"  Weights: degree={w['degree']}, betweenness={w['betweenness']}, "
          f"closeness={w['closeness']}")

    # Ranking 1: All-network
    centrality_all = centrality_all.sort_values("combined_score", ascending=False)
    centrality_all["rank_all"] = range(1, len(centrality_all) + 1)

    all_path = os.path.join(RESULTS_DIR, "protein_centrality_all.csv")
    centrality_all.to_csv(all_path, index=False)
    print(f"\n  All-network centrality saved to: {all_path}")
    print(f"  Total proteins ranked: {len(centrality_all)}")

    # Ranking 2: AML seeds only
    ranking_seeds = centrality_all[centrality_all["node_type"] == "aml_seed"].copy()
    ranking_seeds = ranking_seeds.sort_values("combined_score", ascending=False)
    ranking_seeds["rank"] = range(1, len(ranking_seeds) + 1)

    # Select final output columns
    seed_output_cols = [
        "rank", "protein", "protein_name", "degree", "degree_centrality",
        "betweenness", "closeness", "combined_score",
    ]
    ranking_seeds_out = ranking_seeds[seed_output_cols].copy()

    seed_path = os.path.join(RESULTS_DIR, "final_protein_ranking.csv")
    ranking_seeds_out.to_csv(seed_path, index=False)
    print(f"  AML seed protein ranking saved to: {seed_path}")
    print(f"  AML seed proteins ranked: {len(ranking_seeds_out)}")

    # Print top candidates
    top_n = min(TOP_N_PROTEINS, len(ranking_seeds_out))
    if top_n > 0:
        print(f"\n  Top {top_n} Candidate AML-Associated Proteins:")
        print(f"  {'Rank':<6} {'Protein Name':<15} {'Degree':<8} "
              f"{'Betweenness':<14} {'Closeness':<12} {'Combined':<10}")
        print("  " + "-" * 65)
        for _, row in ranking_seeds_out.head(top_n).iterrows():
            print(f"  {int(row['rank']):<6} {row['protein_name']:<15} "
                  f"{int(row['degree']):<8} {row['betweenness']:<14.6f} "
                  f"{row['closeness']:<12.6f} {row['combined_score']:<10.6f}")

    # --- Visualization: Top proteins bar chart ---
    _generate_ranking_plot(ranking_seeds_out)

    print("\n✓ Centrality analysis & ranking complete.")
    return centrality_all, ranking_seeds_out


def _generate_ranking_plot(ranking_df: pd.DataFrame):
    """Bar chart of top-ranked AML seed proteins by combined score."""
    top = ranking_df.head(min(20, len(ranking_df)))
    if len(top) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(top))

    bars = ax.barh(y_pos, top["combined_score"].values, color="#E74C3C", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["protein_name"].values, fontsize=9)
    ax.invert_yaxis()  # Highest rank on top
    ax.set_xlabel("Combined Centrality Score", fontsize=12)
    ax.set_title("Top Candidate AML-Associated Proteins\n(Ranked by Combined Centrality)",
                 fontsize=13)

    # Add score labels
    for bar, score in zip(bars, top["combined_score"].values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "top_proteins_ranking.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Ranking plot saved to: {fig_path}")


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    from differential_expression import run_differential_expression
    from gene_protein_mapping import map_genes_to_proteins
    from ppi_network import build_ppi_network

    data = load_and_prepare_data()
    _, selected = run_differential_expression(
        data["expression"], data["gene_meta"],
        data["aml_sample_ids"], data["all_sample_ids"],
    )
    mapping_df, mapped_proteins, id_to_name = map_genes_to_proteins(selected)
    G, _, _ = build_ppi_network(mapped_proteins, id_to_name)
    centrality_all, ranking_seeds = compute_centrality_and_rank(G)
