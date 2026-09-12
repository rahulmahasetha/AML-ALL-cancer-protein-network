"""
Master Pipeline: Stage 1 — AML PPI Network Analysis.

Orchestrates all pipeline steps sequentially, collects results,
and prints a comprehensive final summary. All outputs are described
as 'candidate AML-associated proteins' — not proven AML drivers.
"""

import os
import sys
import time

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RESULTS_DIR,
    TOP_N_PROTEINS,
    ADJUSTED_PVALUE_THRESHOLD,
    LOG2FC_THRESHOLD,
    STRING_CONFIDENCE_THRESHOLD,
    CENTRALITY_WEIGHTS,
    DISTANCE_EPSILON,
    N_PERMUTATIONS,
    RANDOM_SEED,
)


def main():
    start_time = time.time()

    print("╔" + "═" * 62 + "╗")
    print("║  STAGE 1: AML PPI NETWORK ANALYSIS PIPELINE                  ║")
    print("║  Candidate AML-Associated Protein Identification             ║")
    print("╚" + "═" * 62 + "╝")
    print()
    print(f"Configuration:")
    print(f"  Random seed:              {RANDOM_SEED}")
    print(f"  Adjusted p-value cutoff:  {ADJUSTED_PVALUE_THRESHOLD}")
    print(f"  |log2FC| cutoff:          {LOG2FC_THRESHOLD}")
    print(f"  STRING confidence cutoff: {STRING_CONFIDENCE_THRESHOLD}")
    print(f"  Centrality weights:       {CENTRALITY_WEIGHTS}")
    print(f"  Distance epsilon:         {DISTANCE_EPSILON}")
    print(f"  Permutations:             {N_PERMUTATIONS}")
    print(f"  Top N proteins:           {TOP_N_PROTEINS}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Data Loading
    # =========================================================================
    from data_loader import load_and_prepare_data
    data = load_and_prepare_data()

    # =========================================================================
    # Steps 2-3: Differential Expression & Gene Selection
    # =========================================================================
    from differential_expression import run_differential_expression
    de_results, selected_genes = run_differential_expression(
        data["expression"],
        data["gene_meta"],
        data["aml_sample_ids"],
        data["all_sample_ids"],
    )

    if len(selected_genes) == 0:
        print("\n⚠ No genes passed the selection criteria. Pipeline cannot continue.")
        print("  Consider relaxing thresholds in config.py.")
        return

    # =========================================================================
    # Step 4: Gene → Protein Mapping
    # =========================================================================
    from gene_protein_mapping import map_genes_to_proteins
    mapping_df, mapped_proteins, id_to_name = map_genes_to_proteins(selected_genes)

    if len(mapped_proteins) == 0:
        print("\n⚠ No genes could be mapped to STRING proteins. Pipeline cannot continue.")
        return

    # =========================================================================
    # Steps 5-6: PPI Network Construction
    # =========================================================================
    from ppi_network import build_ppi_network
    G, edge_df, network_stats = build_ppi_network(mapped_proteins, id_to_name)

    if G.number_of_nodes() == 0:
        print("\n⚠ PPI network is empty. Pipeline cannot continue.")
        return

    # =========================================================================
    # Steps 7-8: Centrality Analysis & Ranking
    # =========================================================================
    from network_analysis import compute_centrality_and_rank
    centrality_all, ranking_seeds = compute_centrality_and_rank(G)

    # =========================================================================
    # Step 9: Permutation Test
    # =========================================================================
    from permutation_test import run_permutation_test
    n_seeds = (centrality_all["node_type"] == "aml_seed").sum()
    perm_results = run_permutation_test(centrality_all, n_seeds)

    # =========================================================================
    # Step 10: Functional Enrichment
    # =========================================================================
    from enrichment import run_enrichment
    enrichment_df = run_enrichment(ranking_seeds, mapped_proteins, id_to_name)

    # =========================================================================
    # Step 11: Independent Validation
    # =========================================================================
    from validation import run_independent_validation
    validation_df = run_independent_validation(selected_genes)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    _print_final_summary(
        data, de_results, selected_genes, mapping_df, mapped_proteins,
        network_stats, ranking_seeds, perm_results, enrichment_df,
        validation_df, elapsed,
    )


def _print_final_summary(
    data, de_results, selected_genes, mapping_df, mapped_proteins,
    network_stats, ranking_seeds, perm_results, enrichment_df,
    validation_df, elapsed,
):
    """Print comprehensive final summary."""
    print("\n")
    print("╔" + "═" * 62 + "╗")
    print("║                    FINAL SUMMARY                             ║")
    print("╚" + "═" * 62 + "╝")

    print(f"\n  ── Data ──")
    print(f"  Total genes analysed:        {len(de_results)}")
    print(f"  AML samples:                 {len(data['aml_sample_ids'])}")
    print(f"  ALL samples:                 {len(data['all_sample_ids'])}")

    print(f"\n  ── Differential Expression ──")
    sig_genes = (de_results["adjusted_p_value"] < ADJUSTED_PVALUE_THRESHOLD).sum()
    print(f"  Significant genes (adj p < {ADJUSTED_PVALUE_THRESHOLD}): {sig_genes}")
    print(f"  Selected genes (+ |log2FC| > {LOG2FC_THRESHOLD}): {len(selected_genes)}")

    print(f"\n  ── Gene → Protein Mapping ──")
    n_mapped = (mapping_df["mapping_status"] == "mapped").sum()
    n_total = len(mapping_df)
    pct = round(100 * n_mapped / n_total, 1) if n_total > 0 else 0
    print(f"  Successfully mapped:         {n_mapped} / {n_total} ({pct}%)")
    print(f"  Unique STRING proteins:      {len(mapped_proteins)}")

    print(f"\n  ── PPI Network ──")
    print(f"  Total nodes:                 {network_stats['total_nodes']}")
    print(f"  Total edges:                 {network_stats['total_edges']}")
    print(f"  AML seed nodes:              {network_stats['aml_seed_nodes']}")
    print(f"  1-hop neighbor nodes:        {network_stats['neighbor_nodes']}")
    print(f"  Connected components:        {network_stats['connected_components']}")
    print(f"  Network density:             {network_stats['density']}")

    print(f"\n  ── Top {min(20, len(ranking_seeds))} Candidate AML-Associated Proteins ──")
    for _, row in ranking_seeds.head(20).iterrows():
        print(f"    #{int(row['rank']):<4d} {row['protein_name']:<15s}  "
              f"combined={row['combined_score']:.4f}  degree={int(row['degree'])}")

    print(f"\n  ── Permutation Test ──")
    if perm_results and not (isinstance(perm_results.get("p_value"), float)
                             and perm_results["p_value"] != perm_results["p_value"]):
        print(f"  Observed mean centrality:    {perm_results.get('observed_mean_centrality', 'N/A')}")
        print(f"  Empirical p-value:           {perm_results.get('empirical_p_value', 'N/A')}")
        p = perm_results.get("empirical_p_value", 1.0)
        if p < 0.05:
            print(f"  → SIGNIFICANT: AML seed proteins have higher centrality than chance.")
        else:
            print(f"  → Not significant at p < 0.05 level.")

    print(f"\n  ── Enrichment ──")
    if enrichment_df is not None and len(enrichment_df) > 0:
        sig_enrich = enrichment_df[enrichment_df["fdr"].astype(float) < 0.05] if "fdr" in enrichment_df.columns else pd.DataFrame()
        print(f"  Total enriched terms:        {len(enrichment_df)}")
        print(f"  Significant (FDR < 0.05):    {len(sig_enrich)}")
        if len(sig_enrich) > 0:
            top3 = sig_enrich.head(3)
            print(f"  Top enriched pathways:")
            for _, row in top3.iterrows():
                print(f"    • {row['description'][:55]}")
    else:
        print(f"  Enrichment not available (API call failed or skipped).")

    print(f"\n  ── Independent Validation ──")
    if validation_df is not None and len(validation_df) > 0:
        validated = validation_df[validation_df["status"] == "validated"]
        if len(validated) > 0:
            n_consistent = validated["direction_consistent"].sum()
            pct_consistent = round(100 * n_consistent / len(validated), 1)
            print(f"  Genes validated:             {len(validated)}")
            print(f"  Direction-consistent:         {int(n_consistent)} "
                  f"({pct_consistent}%)")
        else:
            print(f"  No genes could be validated.")
    else:
        print(f"  Independent dataset not available – validation skipped.")

    print(f"\n  ── Output Files ──")
    result_files = sorted(os.listdir(RESULTS_DIR)) if os.path.isdir(RESULTS_DIR) else []
    for f in result_files:
        fpath = os.path.join(RESULTS_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:<45s} ({size_kb:.1f} KB)")

    print(f"\n  Pipeline completed in {elapsed:.1f} seconds.")

    print("\n" + "═" * 64)
    print("  NOTE: All results are CANDIDATE AML-associated proteins")
    print("  identified through statistical association and network")
    print("  centrality — not experimentally validated AML drivers.")
    print("═" * 64)
    print()


if __name__ == "__main__":
    main()
