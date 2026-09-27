"""
Steps 2-3: Differential Expression Analysis & AML/ALL Gene Selection.

Compares AML vs ALL expression for every gene using Welch's t-test,
applies Benjamini-Hochberg FDR correction, selects genes meeting
the configured significance and effect-size thresholds, and generates
a volcano plot for both AML-enriched and ALL-enriched genes.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ADJUSTED_PVALUE_THRESHOLD,
    LOG2FC_THRESHOLD,
    FOLD_CHANGE_PSEUDOCOUNT,
    RESULTS_DIR,
)


def run_differential_expression(expression, gene_meta, aml_ids, all_ids):
    """
    Compare AML vs ALL expression for every gene.

    Parameters
    ----------
    expression : pd.DataFrame
        Genes × samples, columns are integer sample IDs.
    gene_meta : pd.DataFrame
        Columns: gene_description, gene_accession.
    aml_ids : list[int]
        Sample IDs labelled AML.
    all_ids : list[int]
        Sample IDs labelled ALL.

    Returns
    -------
    de_results : pd.DataFrame
        Full differential-expression results for all genes.
    selected_genes : pd.DataFrame
        Subset passing significance + effect-size thresholds for both AML and ALL.
    """
    print("\n" + "=" * 60)
    print("STEP 2: DIFFERENTIAL EXPRESSION ANALYSIS")
    print("=" * 60)

    aml_expr = expression[aml_ids]
    all_expr = expression[all_ids]

    results = []
    epsilon = FOLD_CHANGE_PSEUDOCOUNT

    for idx in range(len(expression)):
        aml_vals = aml_expr.iloc[idx].dropna().values.astype(float)
        all_vals = all_expr.iloc[idx].dropna().values.astype(float)

        aml_mean = np.mean(aml_vals) if len(aml_vals) > 0 else np.nan
        all_mean = np.mean(all_vals) if len(all_vals) > 0 else np.nan

        # Log2 fold-change with shift to handle negative microarray values.
        # Microarray data can have negative values after background subtraction.
        # We shift both means so the smaller is at least epsilon, preserving
        # the ratio direction while keeping all values positive for log2.
        shift = max(0, epsilon - min(aml_mean, all_mean))
        aml_shifted = aml_mean + shift + epsilon
        all_shifted = all_mean + shift + epsilon
        log2fc = np.log2(aml_shifted) - np.log2(all_shifted)

        # Welch's t-test (unequal variance)
        if len(aml_vals) >= 2 and len(all_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(aml_vals, all_vals, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan

        results.append({
            "gene": gene_meta.iloc[idx]["gene_accession"],
            "gene_description": gene_meta.iloc[idx]["gene_description"],
            "AML_mean": round(aml_mean, 4),
            "ALL_mean": round(all_mean, 4),
            "log2_fold_change": round(log2fc, 6),
            "t_statistic": round(t_stat, 6) if not np.isnan(t_stat) else np.nan,
            "p_value": p_val,
        })

    de_results = pd.DataFrame(results)

    # --- FDR correction (Benjamini-Hochberg) ---
    valid_mask = de_results["p_value"].notna()
    adjusted = np.full(len(de_results), np.nan)
    if valid_mask.sum() > 0:
        _, adj_pvals, _, _ = multipletests(
            de_results.loc[valid_mask, "p_value"].values,
            method="fdr_bh",
        )
        adjusted[valid_mask.values] = adj_pvals
    de_results["adjusted_p_value"] = adjusted

    # Sort by adjusted p-value
    de_results = de_results.sort_values("adjusted_p_value").reset_index(drop=True)

    # Save full results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    de_path = os.path.join(RESULTS_DIR, "differential_expression.csv")
    de_results.to_csv(de_path, index=False)
    print(f"\n  Full DE results saved to: {de_path}")
    print(f"  Total genes analysed: {len(de_results)}")
    print(f"  Genes with valid p-value: {valid_mask.sum()}")

    # --- Step 3: Gene selection ---
    print("\n" + "=" * 60)
    print("STEP 3: CANDIDATE AML/ALL-ASSOCIATED GENE SELECTION")
    print("=" * 60)
    print(f"  Criteria: adjusted_p_value < {ADJUSTED_PVALUE_THRESHOLD} ")
    print(f"  AML-enriched: log2FC > +{LOG2FC_THRESHOLD}")
    print(f"  ALL-enriched: log2FC < -{LOG2FC_THRESHOLD}")

    # Add class_association column
    de_results["class_association"] = "non-significant"
    de_results.loc[
        (de_results["adjusted_p_value"] < ADJUSTED_PVALUE_THRESHOLD) & 
        (de_results["log2_fold_change"] > LOG2FC_THRESHOLD), 
        "class_association"
    ] = "AML-enriched"
    
    de_results.loc[
        (de_results["adjusted_p_value"] < ADJUSTED_PVALUE_THRESHOLD) & 
        (de_results["log2_fold_change"] < -LOG2FC_THRESHOLD), 
        "class_association"
    ] = "ALL-enriched"

    selected = de_results[de_results["class_association"].isin(["AML-enriched", "ALL-enriched"])].copy()
    selected = selected.sort_values(by=["adjusted_p_value", "log2_fold_change"], ascending=[True, False]).reset_index(drop=True)
    selected["rank"] = range(1, len(selected) + 1)
    selected["effect_size"] = selected["log2_fold_change"]

    sel_path = os.path.join(RESULTS_DIR, "selected_differential_genes.csv")
    selected.to_csv(sel_path, index=False)
    aml_count = (selected['class_association'] == 'AML-enriched').sum()
    all_count = (selected['class_association'] == 'ALL-enriched').sum()
    print(f"  Selected genes: {len(selected)}")
    print(f"    AML-enriched: {aml_count}")
    print(f"    ALL-enriched: {all_count}")
    print(f"  Saved to: {sel_path}")

    if len(selected) > 0:
        print(f"\n  Top 10 selected genes:")
        for _, row in selected.head(10).iterrows():
            print(f"    {row['gene']:25s}  log2FC={row['log2_fold_change']:+.3f}  "
                  f"adj_p={row['adjusted_p_value']:.2e}  [{row['class_association']}]")

    # --- Volcano plot ---
    _generate_volcano_plot(de_results)

    print("\n✓ Differential expression & gene selection complete.")
    return de_results, selected


def _generate_volcano_plot(de_results: pd.DataFrame):
    """Generate a volcano plot: log2FC vs -log10(adjusted p-value)."""
    df = de_results.dropna(subset=["adjusted_p_value", "log2_fold_change"]).copy()
    df["neg_log10_adj_p"] = -np.log10(df["adjusted_p_value"].clip(lower=1e-300))

    # Classify points
    aml_mask = (df["adjusted_p_value"] < ADJUSTED_PVALUE_THRESHOLD) & (df["log2_fold_change"] > LOG2FC_THRESHOLD)
    all_mask = (df["adjusted_p_value"] < ADJUSTED_PVALUE_THRESHOLD) & (df["log2_fold_change"] < -LOG2FC_THRESHOLD)
    nonsig_mask = ~(aml_mask | all_mask)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Non-significant
    ax.scatter(
        df.loc[nonsig_mask, "log2_fold_change"],
        df.loc[nonsig_mask, "neg_log10_adj_p"],
        c="grey", alpha=0.4, s=8, label="Not significant",
    )
    # AML-enriched
    ax.scatter(
        df.loc[aml_mask, "log2_fold_change"],
        df.loc[aml_mask, "neg_log10_adj_p"],
        c="crimson", alpha=0.7, s=15, label="AML-enriched",
    )
    # ALL-enriched
    ax.scatter(
        df.loc[all_mask, "log2_fold_change"],
        df.loc[all_mask, "neg_log10_adj_p"],
        c="blue", alpha=0.7, s=15, label="ALL-enriched",
    )

    # Threshold lines
    ax.axhline(-np.log10(ADJUSTED_PVALUE_THRESHOLD), ls="--", c="green", alpha=0.5,
               label=f"adj. p = {ADJUSTED_PVALUE_THRESHOLD}")
    ax.axvline(-LOG2FC_THRESHOLD, ls="--", c="purple", alpha=0.5,
               label=f"log2FC = -{LOG2FC_THRESHOLD}")
    ax.axvline(LOG2FC_THRESHOLD, ls="--", c="orange", alpha=0.5,
               label=f"log2FC = {LOG2FC_THRESHOLD}")

    ax.set_xlabel("log₂ Fold-Change (AML / ALL)", fontsize=12)
    ax.set_ylabel("-log₁₀ Adjusted P-value", fontsize=12)
    ax.set_title("Volcano Plot: Differential Expression (AML vs ALL)", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "volcano_plot.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Volcano plot saved to: {fig_path}")


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    data = load_and_prepare_data()
    de_results, selected = run_differential_expression(
        data["expression"], data["gene_meta"],
        data["aml_sample_ids"], data["all_sample_ids"],
    )
