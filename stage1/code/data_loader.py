"""
Step 1: Data Loading and Cleaning.

Loads the Golub AML/ALL gene-expression dataset and sample labels,
separates metadata from expression values, validates sample-label
matching, and reports dataset statistics.
"""

import os
import re
import sys
import pandas as pd
import numpy as np

# Allow running as script or module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_EXPRESSION_FILE,
    LABELS_FILE,
    RESULTS_DIR,
)


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load actual.csv and return a DataFrame with columns [patient, cancer]."""
    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip().lower() for c in labels.columns]
    assert "patient" in labels.columns and "cancer" in labels.columns, (
        f"Expected columns 'patient' and 'cancer' in {labels_path}, "
        f"got {list(labels.columns)}"
    )
    labels["patient"] = labels["patient"].astype(int)
    labels["cancer"] = labels["cancer"].str.strip().str.upper()
    return labels


def load_expression_data(expr_path: str):
    """
    Load the Golub gene-expression CSV.

    The CSV has an interleaved column layout:
      Gene Description, Gene Accession Number, <sample_id>, call, <sample_id>, call, ...

    Returns
    -------
    gene_meta : pd.DataFrame
        Columns: gene_description, gene_accession
    expression : pd.DataFrame
        Rows = genes, columns = sample IDs (int), values = expression
    call_df : pd.DataFrame
        Rows = genes, columns = sample IDs (int), values = A/P/M call strings
    """
    raw = pd.read_csv(expr_path)

    # --- Identify metadata columns ---
    meta_cols = ["Gene Description", "Gene Accession Number"]
    for c in meta_cols:
        assert c in raw.columns, f"Missing expected column '{c}' in expression data"

    gene_meta = raw[meta_cols].copy()
    gene_meta.columns = ["gene_description", "gene_accession"]

    # --- Separate expression and call columns ---
    # NOTE: pandas auto-renames duplicate 'call' columns to 'call.1',
    # 'call.2', etc. We match these using a regex pattern.
    remaining_cols = [c for c in raw.columns if c not in meta_cols]
    _call_pattern = re.compile(r"^call(\.\d+)?$", re.IGNORECASE)

    sample_ids = []
    expr_col_names = []
    call_col_names = []

    i = 0
    while i < len(remaining_cols):
        col = remaining_cols[i]
        # Check if this is a 'call' or 'call.N' column
        if _call_pattern.match(col):
            call_col_names.append(col)
            i += 1
            continue
        try:
            sid = int(col)
            sample_ids.append(sid)
            expr_col_names.append(col)
            # The next column should be a call column
            if (i + 1 < len(remaining_cols)
                    and _call_pattern.match(remaining_cols[i + 1])):
                call_col_names.append(remaining_cols[i + 1])
                i += 2
            else:
                i += 1
        except ValueError:
            # Unknown non-numeric, non-call column – skip
            print(f"  [WARNING] Skipping unexpected column: '{col}'")
            i += 1

    # Build expression DataFrame with integer sample IDs as columns
    expression = pd.DataFrame(index=range(len(raw)))
    for sid, ecol in zip(sample_ids, expr_col_names):
        expression[sid] = pd.to_numeric(raw[ecol], errors="coerce")

    # Build call DataFrame
    call_df = pd.DataFrame(index=range(len(raw)))
    for sid, ccol in zip(sample_ids, call_col_names):
        call_df[sid] = raw[ccol].values if ccol in raw.columns else np.nan

    return gene_meta, expression, call_df


def match_samples_with_labels(expression: pd.DataFrame, labels: pd.DataFrame):
    """
    Match expression sample IDs with labels from actual.csv.

    Returns
    -------
    matched_labels : pd.DataFrame
        Subset of labels for samples present in the expression matrix.
    aml_sample_ids : list[int]
    all_sample_ids : list[int]

    Raises
    ------
    AssertionError if any expression sample has no matching label.
    """
    expr_sample_ids = set(expression.columns.tolist())
    label_patient_ids = set(labels["patient"].tolist())

    # Every expression sample must have a label
    missing_labels = expr_sample_ids - label_patient_ids
    assert len(missing_labels) == 0, (
        f"Expression samples without labels: {sorted(missing_labels)}"
    )

    matched_labels = labels[labels["patient"].isin(expr_sample_ids)].copy()
    matched_labels = matched_labels.sort_values("patient").reset_index(drop=True)

    aml_sample_ids = matched_labels.loc[
        matched_labels["cancer"] == "AML", "patient"
    ].tolist()
    all_sample_ids = matched_labels.loc[
        matched_labels["cancer"] == "ALL", "patient"
    ].tolist()

    return matched_labels, aml_sample_ids, all_sample_ids


def compute_missing_stats(expression: pd.DataFrame) -> dict:
    """Compute missing-value statistics for the expression matrix."""
    total_cells = expression.size
    missing_cells = expression.isna().sum().sum()
    return {
        "total_cells": int(total_cells),
        "missing_cells": int(missing_cells),
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells > 0 else 0.0,
        "genes_with_any_missing": int(expression.isna().any(axis=1).sum()),
    }


def load_and_prepare_data():
    """
    Master function: load, clean, match, and report.

    Returns
    -------
    dict with keys:
        gene_meta, expression, call_df, labels,
        aml_sample_ids, all_sample_ids, missing_stats
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 60)

    # --- Load labels ---
    print(f"\nLoading labels from: {LABELS_FILE}")
    labels = load_labels(LABELS_FILE)
    print(f"  Total patients in labels file: {len(labels)}")
    print(f"  Label distribution (all patients):")
    for cancer_type, count in labels["cancer"].value_counts().items():
        print(f"    {cancer_type}: {count}")

    # --- Load expression ---
    print(f"\nLoading expression data from: {TRAIN_EXPRESSION_FILE}")
    gene_meta, expression, call_df = load_expression_data(TRAIN_EXPRESSION_FILE)
    print(f"  Number of genes: {len(gene_meta)}")
    print(f"  Number of samples: {len(expression.columns)}")
    print(f"  Sample IDs: {sorted(expression.columns.tolist())}")

    # --- Match samples with labels (NO hard-coded counts) ---
    print("\nMatching expression samples with labels...")
    matched_labels, aml_sample_ids, all_sample_ids = match_samples_with_labels(
        expression, labels
    )

    # Dynamically determined counts
    n_aml = len(aml_sample_ids)
    n_all = len(all_sample_ids)
    print(f"  AML samples (verified from actual.csv): {n_aml}")
    print(f"    IDs: {sorted(aml_sample_ids)}")
    print(f"  ALL samples (verified from actual.csv): {n_all}")
    print(f"    IDs: {sorted(all_sample_ids)}")
    assert n_aml > 0, "No AML samples found – check actual.csv"
    assert n_all > 0, "No ALL samples found – check actual.csv"

    # --- Missing-value stats ---
    missing_stats = compute_missing_stats(expression)
    print(f"\nMissing-value statistics:")
    print(f"  Total cells: {missing_stats['total_cells']}")
    print(f"  Missing cells: {missing_stats['missing_cells']} ({missing_stats['missing_pct']}%)")
    print(f"  Genes with any missing: {missing_stats['genes_with_any_missing']}")

    print("\n✓ Data loading complete.")
    return {
        "gene_meta": gene_meta,
        "expression": expression,
        "call_df": call_df,
        "labels": matched_labels,
        "aml_sample_ids": aml_sample_ids,
        "all_sample_ids": all_sample_ids,
        "missing_stats": missing_stats,
    }


if __name__ == "__main__":
    data = load_and_prepare_data()
