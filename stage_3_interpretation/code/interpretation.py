import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_interpretation(top_n_gnn=50, top_n_stage1=50, consensus_threshold=0.15):
    print("\n" + "=" * 60)
    print("STAGE 3: INTERPRETATION (Consensus Biomarker Discovery)")
    print("=" * 60)
    
    stage2_important_path = os.path.join("stage2", "results", "important_proteins.csv")
    if not os.path.exists(stage2_important_path):
        print(f"Error: {stage2_important_path} not found. Ensure Stage 2 ran successfully.")
        return
        
    gnn_df = pd.read_csv(stage2_important_path)
    
    # 1. Normalize GNN Importance to [0,1]
    gnn_df['S_gnn'] = (gnn_df['importance_score'] - gnn_df['importance_score'].min()) / (gnn_df['importance_score'].max() - gnn_df['importance_score'].min())
    
    stage1_dir = os.path.join("outputs", "stage1")
    comb_path = os.path.join(stage1_dir, "protein_network_ranking.csv")
    if not os.path.exists(comb_path):
        print(f"Error: {comb_path} not found. Ensure Stage 1 ran successfully.")
        return
        
    stage1_df = pd.read_csv(comb_path)
    # 2. Stage 1 Centrality is already min-max normalized as 'combined_score' in [0,1]
    # But let's guarantee it's strictly [0,1] just in case
    stage1_df['S_centrality'] = (stage1_df['combined_score'] - stage1_df['combined_score'].min()) / (stage1_df['combined_score'].max() - stage1_df['combined_score'].min())
    
    # Merge on protein_id
    # Use inner merge since we only care about proteins that are part of the network and have GNN scores
    consensus_df = pd.merge(
        stage1_df[['protein_id', 'protein_name', 'association_group', 'S_centrality']], 
        gnn_df[['protein_id', 'S_gnn', 'importance_score']], 
        on='protein_id', 
        how='inner'
    )
    
    # 3. Calculate Consensus Score
    consensus_df['S_consensus'] = 0.5 * consensus_df['S_centrality'] + 0.5 * consensus_df['S_gnn']
    
    # 4. Filter by consensus threshold rather than fixed Top-N
    consensus_filtered = consensus_df[consensus_df['S_consensus'] >= consensus_threshold].copy()
    
    # 9. Sort descending by consensus score
    consensus_filtered.sort_values(by='S_consensus', ascending=False, inplace=True)
    
    os.makedirs(os.path.join("outputs", "stage3"), exist_ok=True)
    
    # Separate into AML-enriched and ALL-enriched
    aml_consensus = consensus_filtered[consensus_filtered['association_group'] == 'AML-enriched'].copy()
    all_consensus = consensus_filtered[consensus_filtered['association_group'] == 'ALL-enriched'].copy()
    
    # Save the files
    consensus_filtered.to_csv(os.path.join("outputs", "stage3", "consensus_biomarkers.csv"), index=False)
    aml_consensus.to_csv(os.path.join("outputs", "stage3", "consensus_aml_enriched.csv"), index=False)
    all_consensus.to_csv(os.path.join("outputs", "stage3", "consensus_all_enriched.csv"), index=False)
    
    # Print results
    print(f"Selected Threshold for S_consensus: {consensus_threshold}")
    print(f"Total Consensus Biomarkers Discovered: {len(consensus_filtered)}")
    print(f"  - AML-enriched consensus proteins: {len(aml_consensus)}")
    print(f"  - ALL-enriched consensus proteins: {len(all_consensus)}")
    
    print("\nTop 10 Consensus Proteins:")
    print(consensus_filtered[['protein_name', 'association_group', 'S_centrality', 'S_gnn', 'S_consensus']].head(10).to_string(index=False))
    
    # Visualization: Scatter Plot of S_centrality vs S_gnn for selected proteins
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ["#E74C3C" if g == "AML-enriched" else "#2ECC71" for g in consensus_filtered["association_group"]]
    ax.scatter(consensus_filtered["S_centrality"], consensus_filtered["S_gnn"], c=colors, s=50, alpha=0.7)
    
    # Label top few
    for i, row in consensus_filtered.head(10).iterrows():
        ax.annotate(row['protein_name'], (row['S_centrality'], row['S_gnn']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
                    
    # Draw threshold line
    # 0.5 * x + 0.5 * y = threshold => y = 2*threshold - x
    x_vals = np.linspace(0, 1, 100)
    y_vals = 2 * consensus_threshold - x_vals
    ax.plot(x_vals, y_vals, 'k--', label=f'Threshold = {consensus_threshold}')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    ax.set_xlabel("S_centrality (Normalized Topologic Importance)")
    ax.set_ylabel("S_gnn (Normalized GNN Predictive Importance)")
    ax.set_title(f"Consensus Biomarkers Selection (Threshold ≥ {consensus_threshold})")
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=10, label="AML-Enriched"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', markersize=10, label="ALL-Enriched"),
        Line2D([0], [0], color='k', linestyle='--', label=f"S_consensus = {consensus_threshold}")
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    fig.savefig(os.path.join("outputs", "stage3", "consensus_scatter.png"), dpi=150)
    plt.close(fig)
    print("\nConsensus Scatter Plot saved to outputs/stage3/consensus_scatter.png")
    
    # Secondary Analysis: The old Fixed Top-N Intersection
    print("\n" + "-" * 40)
    print(f"Secondary Analysis: Traditional Top-{top_n_stage1} ∩ Top-{top_n_gnn} Overlap")
    
    gnn_top = gnn_df.head(top_n_gnn).copy()
    gnn_proteins = set(gnn_top["protein_id"].tolist())
    
    top_stage1_df = stage1_df.head(top_n_stage1)
    top_proteins = set(top_stage1_df["protein_id"].tolist())
    
    overlap_proteins = gnn_proteins.intersection(top_proteins)
    
    print(f"  Fixed Top-N Intersection yielded {len(overlap_proteins)} proteins.")
    
    # Write summary file
    summary = {
        "consensus_threshold": consensus_threshold,
        "total_consensus_selected": len(consensus_filtered),
        "aml_consensus": len(aml_consensus),
        "all_consensus": len(all_consensus),
        "fixed_top_n_overlap": len(overlap_proteins)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join("outputs", "stage3", "interpretation_summary.csv"), index=False)
    
    print("\n✓ Stage 3 Interpretation complete.")
    
if __name__ == "__main__":
    run_interpretation(top_n_gnn=50, top_n_stage1=50, consensus_threshold=0.15)
