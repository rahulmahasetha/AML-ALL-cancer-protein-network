"""
Step 9: Permutation-Based Statistical Validation.

Tests whether the observed mean combined centrality of AML seed proteins
is significantly higher than expected by chance, using a permutation test
with the corrected p-value formula: (count + 1) / (N + 1).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import N_PERMUTATIONS, RANDOM_SEED, RESULTS_DIR


def run_permutation_test(centrality_all: pd.DataFrame, n_aml: int, n_all: int):
    """
    Permutation test for AML/ALL seed protein centrality enrichment.

    Null hypothesis: The mean combined centrality of seed proteins
    is not higher than expected from a random set of the same size
    drawn from all proteins in the network.

    Parameters
    ----------
    centrality_all : pd.DataFrame
        Must have columns: protein, node_type, combined_score.
    n_aml : int
        Number of AML-enriched seed proteins.
    n_all : int
        Number of ALL-enriched seed proteins.

    Returns
    -------
    perm_results : dict
        Observed statistic, null distribution summary, empirical p-values for AML, ALL, combined.
    """
    print("\n" + "=" * 60)
    print("STEP 9: PERMUTATION TEST")
    print("=" * 60)

    n_combined = n_aml + n_all
    if len(centrality_all) == 0 or n_combined == 0:
        print("  ⚠ No data for permutation test – skipping.")
        return {"p_value": np.nan, "observed_stat": np.nan}

    all_scores = centrality_all["combined_score"].values
    aml_scores = centrality_all.loc[centrality_all["node_type"] == "aml_seed", "combined_score"].values
    all_seed_scores = centrality_all.loc[centrality_all["node_type"] == "all_seed", "combined_score"].values
    combined_scores = np.concatenate([aml_scores, all_seed_scores]) if len(aml_scores) > 0 or len(all_seed_scores) > 0 else np.array([])

    total_proteins = len(all_scores)

    def permute_group(scores, n_size, group_name):
        if n_size == 0 or len(scores) == 0:
            return np.nan, np.nan, np.nan, np.nan
        obs_stat = np.mean(scores)
        rng = np.random.RandomState(RANDOM_SEED)
        null_stats = np.zeros(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            random_indices = rng.choice(total_proteins, size=n_size, replace=False)
            null_stats[i] = np.mean(all_scores[random_indices])
        count_extreme = np.sum(null_stats >= obs_stat)
        p_val = (count_extreme + 1) / (N_PERMUTATIONS + 1)
        print(f"\n  {group_name} Permutation results:")
        print(f"    Observed mean centrality: {obs_stat:.6f}")
        print(f"    Empirical p-value: {p_val:.6f}")
        return obs_stat, np.mean(null_stats), np.std(null_stats), p_val

    print(f"\n  Number of permutations: {N_PERMUTATIONS}")

    aml_obs, aml_null_mean, aml_null_std, aml_p = permute_group(aml_scores, n_aml, "AML-enriched")
    all_obs, all_null_mean, all_null_std, all_p = permute_group(all_seed_scores, n_all, "ALL-enriched")
    comb_obs, comb_null_mean, comb_null_std, comb_p = permute_group(combined_scores, n_combined, "Combined")

    perm_results = {
        "n_permutations": N_PERMUTATIONS,
        "total_network_proteins": total_proteins,
        "aml_observed_mean": round(aml_obs, 6) if pd.notna(aml_obs) else np.nan,
        "aml_p_value": round(aml_p, 6) if pd.notna(aml_p) else np.nan,
        "all_observed_mean": round(all_obs, 6) if pd.notna(all_obs) else np.nan,
        "all_p_value": round(all_p, 6) if pd.notna(all_p) else np.nan,
        "combined_observed_mean": round(comb_obs, 6) if pd.notna(comb_obs) else np.nan,
        "combined_p_value": round(comb_p, 6) if pd.notna(comb_p) else np.nan,
        "random_seed": RANDOM_SEED,
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    perm_path = os.path.join(RESULTS_DIR, "permutation_results.csv")
    pd.DataFrame([perm_results]).to_csv(perm_path, index=False)
    print(f"\n  Permutation results saved to: {perm_path}")

    # Generate null distribution plot for combined
    if pd.notna(comb_obs):
        rng = np.random.RandomState(RANDOM_SEED)
        null_stats = np.zeros(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            random_indices = rng.choice(total_proteins, size=n_combined, replace=False)
            null_stats[i] = np.mean(all_scores[random_indices])
        _generate_null_distribution_plot(null_stats, comb_obs, comb_p, "Combined")

    print("\n✓ Permutation test complete.")
    return perm_results


def _generate_null_distribution_plot(null_stats, observed_stat, p_value, group_name):
    """Histogram of null distribution with observed statistic marked."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(null_stats, bins=50, color="#3498DB", alpha=0.7, edgecolor="white",
            label="Null distribution")
    ax.axvline(observed_stat, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Observed ({group_name}) = {observed_stat:.4f}")
    ax.axvline(np.mean(null_stats), color="#2ECC71", linewidth=1.5, linestyle=":",
               label=f"Null mean = {np.mean(null_stats):.4f}")

    ax.set_xlabel("Mean Combined Centrality Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Permutation Test: {group_name} Centrality Enrichment\n"
        f"Empirical p = {p_value:.4f} ({N_PERMUTATIONS} permutations)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, f"permutation_null_distribution_{group_name.lower()}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Null distribution plot saved to: {fig_path}")


if __name__ == "__main__":
    results_path = os.path.join(RESULTS_DIR, "protein_centrality_all.csv")
    if os.path.exists(results_path):
        centrality_all = pd.read_csv(results_path)
        n_aml = (centrality_all["node_type"] == "aml_seed").sum()
        n_all = (centrality_all["node_type"] == "all_seed").sum()
        run_permutation_test(centrality_all, n_aml, n_all)
    else:
        print("Run the full pipeline first to generate centrality data.")
