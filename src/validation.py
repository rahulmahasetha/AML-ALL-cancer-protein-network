"""
Step 11: Independent Dataset Validation.

Uses data_set_ALL_AML_independent.csv to validate whether genes
selected from the training set show consistent AML-vs-ALL behaviour
in an independent cohort. This dataset is used ONLY for final
validation — no gene selection, threshold tuning, or model fitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INDEPENDENT_EXPRESSION_FILE,
    LABELS_FILE,
    FOLD_CHANGE_PSEUDOCOUNT,
    RESULTS_DIR,
)


def _load_independent_expression(file_path: str):
    """
    Load the independent expression dataset using the same parsing
    logic as the training data.
    """
    raw = pd.read_csv(file_path)

    meta_cols = ["Gene Description", "Gene Accession Number"]
    gene_meta = raw[meta_cols].copy()
    gene_meta.columns = ["gene_description", "gene_accession"]

    remaining_cols = [c for c in raw.columns if c not in meta_cols]

    sample_ids = []
    expr_data = {}

    i = 0
    while i < len(remaining_cols):
        col = remaining_cols[i]
        try:
            sid = int(col)
            sample_ids.append(sid)
            expr_data[sid] = pd.to_numeric(raw[col], errors="coerce").values
            # Skip the following "call" column
            if (i + 1 < len(remaining_cols)
                    and remaining_cols[i + 1].lower() == "call"):
                i += 2
            else:
                i += 1
        except ValueError:
            i += 1

    expression = pd.DataFrame(expr_data, index=range(len(raw)))
    return gene_meta, expression


def run_independent_validation(selected_genes: pd.DataFrame):
    """
    Validate selected AML genes in the independent dataset.

    Parameters
    ----------
    selected_genes : pd.DataFrame
        Genes selected from the training analysis (must have 'gene' column
        with gene accession identifiers, and 'log2_fold_change' from training).

    Returns
    -------
    validation_df : pd.DataFrame or None
        Per-gene validation results, or None if independent data unavailable.
    """
    print("\n" + "=" * 60)
    print("STEP 11: INDEPENDENT DATASET VALIDATION")
    print("=" * 60)

    # --- Check file availability ---
    if not os.path.exists(INDEPENDENT_EXPRESSION_FILE):
        print(f"\n  ⚠ Independent dataset not found:")
        print(f"    {INDEPENDENT_EXPRESSION_FILE}")
        print(f"  Skipping independent validation.")
        print(f"  To enable: place data_set_ALL_AML_independent.csv in data/")
        return None

    print(f"\n  Loading independent expression data...")
    ind_meta, ind_expression = _load_independent_expression(INDEPENDENT_EXPRESSION_FILE)
    print(f"  Independent samples: {len(ind_expression.columns)}")
    print(f"  Independent genes: {len(ind_meta)}")

    # --- Match with labels ---
    labels = pd.read_csv(LABELS_FILE)
    labels.columns = [c.strip().lower() for c in labels.columns]
    labels["patient"] = labels["patient"].astype(int)
    labels["cancer"] = labels["cancer"].str.strip().str.upper()

    ind_sample_ids = set(ind_expression.columns.tolist())
    ind_labels = labels[labels["patient"].isin(ind_sample_ids)]

    ind_aml_ids = ind_labels.loc[ind_labels["cancer"] == "AML", "patient"].tolist()
    ind_all_ids = ind_labels.loc[ind_labels["cancer"] == "ALL", "patient"].tolist()

    print(f"  Independent AML samples: {len(ind_aml_ids)}")
    print(f"  Independent ALL samples: {len(ind_all_ids)}")

    if len(ind_aml_ids) == 0 or len(ind_all_ids) == 0:
        print("  ⚠ Not enough samples in both groups for validation.")
        return None

    # --- Build gene accession index for independent data ---
    ind_gene_acc = ind_meta["gene_accession"].values
    ind_gene_to_idx = {}
    for idx, acc in enumerate(ind_gene_acc):
        ind_gene_to_idx[acc] = idx

    # --- Validate each selected gene ---
    epsilon = FOLD_CHANGE_PSEUDOCOUNT
    validation_records = []

    for _, row in selected_genes.iterrows():
        gene_acc = row["gene"]
        train_log2fc = row.get("log2_fold_change", row.get("effect_size", np.nan))

        if gene_acc not in ind_gene_to_idx:
            validation_records.append({
                "gene": gene_acc,
                "train_log2fc": train_log2fc,
                "ind_AML_mean": np.nan,
                "ind_ALL_mean": np.nan,
                "ind_log2fc": np.nan,
                "direction_consistent": np.nan,
                "ind_p_value": np.nan,
                "status": "not_in_independent",
            })
            continue

        idx = ind_gene_to_idx[gene_acc]
        aml_vals = ind_expression.iloc[idx][ind_aml_ids].dropna().values.astype(float)
        all_vals = ind_expression.iloc[idx][ind_all_ids].dropna().values.astype(float)

        ind_aml_mean = np.mean(aml_vals) if len(aml_vals) > 0 else np.nan
        ind_all_mean = np.mean(all_vals) if len(all_vals) > 0 else np.nan
        
        # Log2 fold-change with shift to handle negative microarray values
        shift = max(0, epsilon - min(ind_aml_mean, ind_all_mean))
        aml_shifted = ind_aml_mean + shift + epsilon
        all_shifted = ind_all_mean + shift + epsilon
        ind_log2fc = np.log2(aml_shifted) - np.log2(all_shifted)

        # Direction consistency
        direction_consistent = (
            (train_log2fc > 0 and ind_log2fc > 0)
            or (train_log2fc < 0 and ind_log2fc < 0)
        ) if not (np.isnan(train_log2fc) or np.isnan(ind_log2fc)) else np.nan

        # P-value in independent set
        if len(aml_vals) >= 2 and len(all_vals) >= 2:
            _, ind_p = stats.ttest_ind(aml_vals, all_vals, equal_var=False)
        else:
            ind_p = np.nan

        validation_records.append({
            "gene": gene_acc,
            "train_log2fc": round(train_log2fc, 6),
            "ind_AML_mean": round(ind_aml_mean, 4),
            "ind_ALL_mean": round(ind_all_mean, 4),
            "ind_log2fc": round(ind_log2fc, 6),
            "direction_consistent": direction_consistent,
            "ind_p_value": ind_p,
            "status": "validated",
        })

    validation_df = pd.DataFrame(validation_records)

    # --- Report ---
    validated = validation_df[validation_df["status"] == "validated"]
    if len(validated) > 0:
        n_consistent = validated["direction_consistent"].sum()
        n_validated = len(validated)
        consistency_pct = round(100 * n_consistent / n_validated, 1)
        n_sig = (validated["ind_p_value"] < 0.05).sum()

        print(f"\n  Validation results:")
        print(f"    Genes found in independent set: {n_validated}")
        print(f"    Direction-consistent:           {int(n_consistent)} / {n_validated} "
              f"({consistency_pct}%)")
        print(f"    Significant in independent set (p<0.05): {n_sig}")
    else:
        print(f"\n  No genes could be validated in the independent dataset.")

    not_found = validation_df[validation_df["status"] == "not_in_independent"]
    if len(not_found) > 0:
        print(f"    Genes not found in independent set: {len(not_found)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_path = os.path.join(RESULTS_DIR, "independent_validation.csv")
    validation_df.to_csv(val_path, index=False)
    print(f"\n  Validation results saved to: {val_path}")

    print("\n✓ Independent validation complete.")
    return validation_df


if __name__ == "__main__":
    sel_path = os.path.join(RESULTS_DIR, "selected_aml_genes.csv")
    if os.path.exists(sel_path):
        selected = pd.read_csv(sel_path)
        run_independent_validation(selected)
    else:
        print("Run the full pipeline first.")
