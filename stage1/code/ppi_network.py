"""
Steps 5-6: STRING PPI Network Loading & NetworkX Graph Construction.

Loads the STRING human PPI data efficiently (streaming), filters by
confidence threshold, builds an AML-seed + 1-hop-neighbor subnetwork,
and reports seed vs neighbour statistics separately.
"""

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STRING_LINKS_FILE,
    STRING_CONFIDENCE_THRESHOLD,
    RESULTS_DIR,
)


def build_ppi_network(mapped_proteins: set, id_to_name: dict, mapping_df: pd.DataFrame):
    """
    Build PPI subnetwork from STRING data.

    Parameters
    ----------
    mapped_proteins : set[str]
        STRING protein IDs that are AML or ALL seed proteins.
    id_to_name : dict[str, str]
        STRING ID → preferred_name.
    mapping_df : pd.DataFrame
        DataFrame with mapped proteins and their class_association.

    Returns
    -------
    G : nx.Graph
        Undirected weighted graph. Nodes have attribute 'node_type'
        (either 'aml_seed', 'all_seed', or 'neighbor') and 'protein_name'.
    edge_df : pd.DataFrame
        Edge table with protein_A, protein_B, combined_score.
    network_stats : dict
        Summary statistics.
    """
    print("\n" + "=" * 60)
    print("STEP 5: STRING PPI NETWORK LOADING")
    print("=" * 60)

    print(f"\n  AML seed proteins: {len(mapped_proteins)}")
    print(f"  STRING confidence threshold: {STRING_CONFIDENCE_THRESHOLD}")
    print(f"  Loading from: {STRING_LINKS_FILE}")

    # --- Memory-efficient streaming: keep edges involving seed proteins ---
    edges = []
    lines_read = 0
    edges_above_threshold = 0

    with open(STRING_LINKS_FILE, "r") as f:
        header = f.readline()  # Skip header
        for line in f:
            lines_read += 1
            if lines_read % 2_000_000 == 0:
                print(f"    ... processed {lines_read:,} lines, kept {len(edges):,} edges")

            parts = line.strip().split()
            if len(parts) < 3:
                continue

            p1, p2, score_str = parts[0], parts[1], parts[2]
            score = int(score_str)

            if score < STRING_CONFIDENCE_THRESHOLD:
                continue
            edges_above_threshold += 1

            # Keep edge if at least one protein is an AML seed
            if p1 in mapped_proteins or p2 in mapped_proteins:
                edges.append((p1, p2, score))

    print(f"\n  Total STRING lines processed: {lines_read:,}")
    print(f"  Edges above threshold ({STRING_CONFIDENCE_THRESHOLD}): {edges_above_threshold:,}")
    print(f"  Edges involving AML seed proteins: {len(edges):,}")

    # --- Build NetworkX graph ---
    print("\n" + "=" * 60)
    print("STEP 6: BUILDING NETWORKX GRAPH")
    print("=" * 60)

    G = nx.Graph()

    # Add edges with weights
    for p1, p2, score in edges:
        G.add_edge(p1, p2, combined_score=score)

    # Build a lookup for class_association
    protein_to_class = {}
    for _, row in mapping_df.iterrows():
        if row["mapping_status"] == "mapped" and pd.notna(row["string_protein_id"]):
            protein_to_class[row["string_protein_id"]] = row.get("class_association", "unknown")

    # Tag nodes as seed or neighbor
    aml_seed_in_network = set()
    all_seed_in_network = set()
    neighbor_nodes = set()
    for node in G.nodes():
        if node in mapped_proteins:
            c_assoc = protein_to_class.get(node, "")
            if c_assoc == "AML-enriched":
                G.nodes[node]["node_type"] = "aml_seed"
                G.nodes[node]["association_group"] = "AML-enriched"
                aml_seed_in_network.add(node)
            elif c_assoc == "ALL-enriched":
                G.nodes[node]["node_type"] = "all_seed"
                G.nodes[node]["association_group"] = "ALL-enriched"
                all_seed_in_network.add(node)
            else:
                G.nodes[node]["node_type"] = "neighbor"
                G.nodes[node]["association_group"] = "neighbor"
                neighbor_nodes.add(node)
        else:
            G.nodes[node]["node_type"] = "neighbor"
            G.nodes[node]["association_group"] = "neighbor"
            neighbor_nodes.add(node)

        # Add protein name
        G.nodes[node]["protein_name"] = id_to_name.get(node, node)
        G.nodes[node]["protein_id"] = node

    # --- Network statistics ---
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    density = nx.density(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0

    # Seed vs neighbor degree stats
    aml_degrees = [d for n, d in G.degree() if G.nodes[n]["node_type"] == "aml_seed"]
    all_degrees = [d for n, d in G.degree() if G.nodes[n]["node_type"] == "all_seed"]
    neighbor_degrees = [d for n, d in G.degree() if G.nodes[n]["node_type"] == "neighbor"]
    
    seed_in_network = aml_seed_in_network | all_seed_in_network

    network_stats = {
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "aml_seed_nodes": len(aml_seed_in_network),
        "all_seed_nodes": len(all_seed_in_network),
        "neighbor_nodes": len(neighbor_nodes),
        "connected_components": n_components,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 2),
        "avg_aml_degree": round(np.mean(aml_degrees), 2) if aml_degrees else 0,
        "avg_all_degree": round(np.mean(all_degrees), 2) if all_degrees else 0,
        "avg_neighbor_degree": round(np.mean(neighbor_degrees), 2) if neighbor_degrees else 0,
        "seeds_not_in_network": len(mapped_proteins - seed_in_network),
    }

    print(f"\n  Network statistics:")
    for k, v in network_stats.items():
        print(f"    {k}: {v}")

    # Seeds with no interactions
    missing_seeds = mapped_proteins - seed_in_network
    if missing_seeds:
        print(f"\n  ⚠ {len(missing_seeds)} seed protein(s) had no interactions "
              f"above threshold and are not in the network.")

    # --- Save edge table ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    edge_records = []
    for p1, p2, data in G.edges(data=True):
        edge_records.append({
            "protein_A": p1,
            "protein_B": p2,
            "combined_score": data["combined_score"],
        })
    edge_df = pd.DataFrame(edge_records)
    edge_path = os.path.join(RESULTS_DIR, "unified_ppi_network_edges.csv")
    edge_df.to_csv(edge_path, index=False)
    print(f"\n  Edge table saved to: {edge_path}")

    # --- Save node table ---
    node_records = []
    for node in G.nodes():
        node_records.append({
            "protein_id": node,
            "protein_name": G.nodes[node].get("protein_name", ""),
            "association_group": G.nodes[node].get("association_group", "neighbor"),
            "node_type": G.nodes[node]["node_type"],
            "degree": G.degree(node),
        })
    node_df = pd.DataFrame(node_records).sort_values("degree", ascending=False)
    node_path = os.path.join(RESULTS_DIR, "ppi_node_metadata.csv")
    node_df.to_csv(node_path, index=False)
    print(f"  Node metadata table saved to: {node_path}")

    # --- Save stats ---
    stats_path = os.path.join(RESULTS_DIR, "network_statistics.csv")
    pd.DataFrame([network_stats]).to_csv(stats_path, index=False)
    print(f"  Network stats saved to: {stats_path}")

    # --- Network visualization ---
    _generate_network_visualization(G, aml_seed_in_network, all_seed_in_network)

    print("\n✓ PPI network construction complete.")
    return G, edge_df, network_stats


def _generate_network_visualization(G: nx.Graph, aml_seed_nodes: set, all_seed_nodes: set):
    """Generate a network visualization with distinct seed/neighbor styles."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # For very large networks, show only the largest connected component
    if G.number_of_nodes() > 2000:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
        title_suffix = f" (Largest Component: {subG.number_of_nodes()} nodes)"
    else:
        subG = G
        title_suffix = ""

    # If still too large for layout, subsample
    if subG.number_of_nodes() > 500:
        # Keep all seeds + top-degree neighbors
        seeds_in_sub = [n for n in subG.nodes() if n in aml_seed_nodes or n in all_seed_nodes]
        neighbors_by_degree = sorted(
            [n for n in subG.nodes() if n not in aml_seed_nodes and n not in all_seed_nodes],
            key=lambda n: subG.degree(n), reverse=True
        )
        keep_nodes = set(seeds_in_sub) | set(neighbors_by_degree[:200])
        subG = subG.subgraph(keep_nodes).copy()
        title_suffix += f" (Top {subG.number_of_nodes()} nodes shown)"

    pos = nx.spring_layout(subG, seed=42, k=1.5 / np.sqrt(max(subG.number_of_nodes(), 1)))

    # Node colours and sizes
    node_colors = []
    node_sizes = []
    for node in subG.nodes():
        if node in aml_seed_nodes:
            node_colors.append("#E74C3C")  # Red for AML seeds
            node_sizes.append(120)
        elif node in all_seed_nodes:
            node_colors.append("#2ECC71")  # Green for ALL seeds
            node_sizes.append(120)
        else:
            node_colors.append("#3498DB")  # Blue for neighbors
            node_sizes.append(30)

    # Draw
    nx.draw_networkx_edges(subG, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.8, ax=ax)

    # Label seed nodes
    seed_labels = {n: subG.nodes[n].get("protein_name", n.split(".")[-1])
                   for n in subG.nodes() if n in aml_seed_nodes or n in all_seed_nodes}
    if len(seed_labels) <= 50:
        nx.draw_networkx_labels(subG, pos, labels=seed_labels, font_size=6,
                                font_weight="bold", ax=ax)

    ax.set_title(f"AML/ALL Unified PPI Subnetwork{title_suffix}", fontsize=14)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markersize=10, label="AML-Enriched Proteins"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ECC71",
               markersize=10, label="ALL-Enriched Proteins"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB",
               markersize=7, label="1-Hop Neighbors"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "unified_ppi_network.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Network visualisation saved to: {fig_path}")


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    from differential_expression import run_differential_expression
    from gene_protein_mapping import map_genes_to_proteins

    data = load_and_prepare_data()
    _, selected = run_differential_expression(
        data["expression"], data["gene_meta"],
        data["aml_sample_ids"], data["all_sample_ids"],
    )
    mapping_df, mapped_proteins, id_to_name = map_genes_to_proteins(selected)
    G, edge_df, stats = build_ppi_network(mapped_proteins, id_to_name)
