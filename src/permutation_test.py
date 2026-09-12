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


def run_permutation_test(centrality_all: pd.DataFrame, n_seeds: int):
    """
    Permutation test for AML seed protein centrality enrichment.

    Null hypothesis: The mean combined centrality of AML seed proteins
    is not higher than expected from a random set of the same size
    drawn from all proteins in the network.

    Parameters
    ----------
    centrality_all : pd.DataFrame
        Must have columns: protein, node_type, combined_score.
    n_seeds : int
        Number of AML seed proteins.

    Returns
    -------
    perm_results : dict
        Observed statistic, null distribution summary, empirical p-value.
    """
    print("\n" + "=" * 60)
    print("STEP 9: PERMUTATION TEST")
    print("=" * 60)

    if len(centrality_all) == 0 or n_seeds == 0:
        print("  ⚠ No data for permutation test – skipping.")
        return {"p_value": np.nan, "observed_stat": np.nan}

    all_scores = centrality_all["combined_score"].values
    seed_mask = centrality_all["node_type"] == "aml_seed"
    seed_scores = centrality_all.loc[seed_mask, "combined_score"].values

    observed_stat = np.mean(seed_scores)
    total_proteins = len(all_scores)

    print(f"\n  Null hypothesis: Mean combined centrality of {n_seeds} AML seed")
    print(f"    proteins is not higher than a random set of {n_seeds} proteins")
    print(f"    drawn from the full network ({total_proteins} proteins).")
    print(f"\n  Observed mean centrality of AML seeds: {observed_stat:.6f}")
    print(f"  Number of permutations: {N_PERMUTATIONS}")

    rng = np.random.RandomState(RANDOM_SEED)
    null_stats = np.zeros(N_PERMUTATIONS)

    for i in range(N_PERMUTATIONS):
        random_indices = rng.choice(total_proteins, size=n_seeds, replace=False)
        null_stats[i] = np.mean(all_scores[random_indices])

    # Corrected empirical p-value: (count(null >= observed) + 1) / (N + 1)
    count_extreme = np.sum(null_stats >= observed_stat)
    p_value = (count_extreme + 1) / (N_PERMUTATIONS + 1)

    null_mean = np.mean(null_stats)
    null_std = np.std(null_stats)

    print(f"\n  Permutation results:")
    print(f"    Null distribution mean: {null_mean:.6f}")
    print(f"    Null distribution std:  {null_std:.6f}")
    print(f"    Null permutations ≥ observed: {count_extreme}")
    print(f"    Empirical p-value: {p_value:.6f}")
    print(f"    Formula: (count({count_extreme}) + 1) / ({N_PERMUTATIONS} + 1) = {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n  → Statistically significant (p < 0.05): AML seed proteins have")
        print(f"    higher combined centrality than expected by chance.")
    else:
        print(f"\n  → Not statistically significant (p ≥ 0.05): AML seed proteins")
        print(f"    do not show significantly higher centrality than random sets.")

    perm_results = {
        "observed_mean_centrality": round(observed_stat, 6),
        "null_mean": round(null_mean, 6),
        "null_std": round(null_std, 6),
        "n_permutations": N_PERMUTATIONS,
        "n_seeds": n_seeds,
        "total_network_proteins": total_proteins,
        "count_null_ge_observed": int(count_extreme),
        "empirical_p_value": round(p_value, 6),
        "random_seed": RANDOM_SEED,
    }

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    perm_path = os.path.join(RESULTS_DIR, "permutation_test.csv")
    pd.DataFrame([perm_results]).to_csv(perm_path, index=False)
    print(f"\n  Permutation results saved to: {perm_path}")

    # Generate null distribution plot
    _generate_null_distribution_plot(null_stats, observed_stat, p_value)

    print("\n✓ Permutation test complete.")
    return perm_results


def _generate_null_distribution_plot(null_stats, observed_stat, p_value):
    """Histogram of null distribution with observed statistic marked."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(null_stats, bins=50, color="#3498DB", alpha=0.7, edgecolor="white",
            label="Null distribution")
    ax.axvline(observed_stat, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Observed (AML seeds) = {observed_stat:.4f}")
    ax.axvline(np.mean(null_stats), color="#2ECC71", linewidth=1.5, linestyle=":",
               label=f"Null mean = {np.mean(null_stats):.4f}")

    ax.set_xlabel("Mean Combined Centrality Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Permutation Test: AML Seed Centrality Enrichment\n"
        f"Empirical p = {p_value:.4f} ({N_PERMUTATIONS} permutations)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "permutation_null_distribution.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Null distribution plot saved to: {fig_path}")


if __name__ == "__main__":
    results_path = os.path.join(RESULTS_DIR, "protein_centrality_all.csv")
    if os.path.exists(results_path):
        centrality_all = pd.read_csv(results_path)
        n_seeds = (centrality_all["node_type"] == "aml_seed").sum()
        run_permutation_test(centrality_all, n_seeds)
    else:
        print("Run the full pipeline first to generate centrality data.")
