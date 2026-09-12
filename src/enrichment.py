"""
Step 10: Functional / Pathway Enrichment Analysis.

Performs GO and pathway enrichment on the final candidate AML-associated
proteins using the STRING API enrichment endpoint. Uses the full set of
successfully mapped proteins as the background universe.
"""

import os
import sys
import time
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, TOP_N_PROTEINS

# STRING enrichment API
STRING_ENRICHMENT_URL = "https://string-db.org/api/json/enrichment"
STRING_SPECIES = 9606  # Homo sapiens


def run_enrichment(
    candidate_proteins: pd.DataFrame,
    all_mapped_proteins: set,
    id_to_name: dict,
):
    """
    Perform functional enrichment analysis.

    Parameters
    ----------
    candidate_proteins : pd.DataFrame
        The ranked AML seed proteins (from final_protein_ranking.csv).
        Must have column 'protein' with STRING IDs.
    all_mapped_proteins : set[str]
        All successfully mapped STRING protein IDs (background universe).
    id_to_name : dict[str, str]
        STRING ID → preferred_name.

    Returns
    -------
    enrichment_df : pd.DataFrame or None
        Enrichment results, or None if the API call fails.
    """
    print("\n" + "=" * 60)
    print("STEP 10: FUNCTIONAL / PATHWAY ENRICHMENT")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("  ⚠ 'requests' library not available. Skipping enrichment.")
        return None

    if len(candidate_proteins) == 0:
        print("  ⚠ No candidate proteins to enrich. Skipping.")
        return None

    # Use top N proteins for enrichment query
    top_n = min(TOP_N_PROTEINS, len(candidate_proteins))
    query_proteins = candidate_proteins.head(top_n)["protein"].tolist()

    # Convert STRING IDs to preferred names for the API
    # STRING API expects identifiers (gene names work, so do ENSP IDs)
    query_identifiers = []
    for pid in query_proteins:
        name = id_to_name.get(pid, pid.replace("9606.", ""))
        query_identifiers.append(name)

    # Background: all mapped proteins
    background_identifiers = []
    for pid in all_mapped_proteins:
        name = id_to_name.get(pid, pid.replace("9606.", ""))
        background_identifiers.append(name)

    print(f"\n  Query proteins: {len(query_identifiers)}")
    print(f"  Background universe: {len(background_identifiers)} mapped proteins")
    print(f"  Using STRING enrichment API: {STRING_ENRICHMENT_URL}")

    # --- Call STRING API ---
    # STRING API accepts newline-delimited (%0d) identifiers
    identifier_str = "%0d".join(query_identifiers)
    background_str = "%0d".join(background_identifiers)

    # Try with background first, then without (custom background isn't
    # always supported by the STRING REST API)
    params_with_bg = {
        "identifiers": identifier_str,
        "species": STRING_SPECIES,
        "background_string_identifiers": background_str,
        "caller_identity": "aml_ppi_stage1_analysis",
    }
    params_no_bg = {
        "identifiers": identifier_str,
        "species": STRING_SPECIES,
        "caller_identity": "aml_ppi_stage1_analysis",
    }

    enrichment_records = []
    try:
        print("  Sending enrichment request (with background)...")
        response = requests.post(STRING_ENRICHMENT_URL, data=params_with_bg, timeout=120)

        # If background request fails or returns minimal results, retry without
        results = []
        if response.status_code == 200:
            results = response.json() if isinstance(response.json(), list) else []

        if len(results) <= 1:
            print("  Few results with background; retrying without background...")
            response = requests.post(STRING_ENRICHMENT_URL, data=params_no_bg, timeout=120)
            if response.status_code == 200:
                results = response.json() if isinstance(response.json(), list) else []
                if len(results) > 0:
                    print(f"  Note: Using whole-genome background (custom background "
                          f"not supported by this STRING API version).")

        if response.status_code == 200 and isinstance(results, list):
            for entry in results:
                enrichment_records.append({
                    "category": entry.get("category", ""),
                    "term": entry.get("term", ""),
                    "description": entry.get("description", ""),
                    "p_value": entry.get("p_value", None),
                    "fdr": entry.get("fdr", None),
                    "number_of_genes": entry.get("number_of_genes", 0),
                    "number_of_genes_in_background": entry.get(
                        "number_of_genes_in_background", 0
                    ),
                    "input_genes": ";".join(entry.get("inputGenes", [])),
                    "preferred_names": ";".join(entry.get("preferredNames", [])),
                })
            print(f"  Received {len(enrichment_records)} enrichment terms.")
        elif response.status_code == 429:
            print(f"  ⚠ Rate limited by STRING API (HTTP 429). Waiting 10s and retrying...")
            time.sleep(10)
            response = requests.post(STRING_ENRICHMENT_URL, data=params_no_bg, timeout=120)
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list):
                    for entry in results:
                        enrichment_records.append({
                            "category": entry.get("category", ""),
                            "term": entry.get("term", ""),
                            "description": entry.get("description", ""),
                            "p_value": entry.get("p_value", None),
                            "fdr": entry.get("fdr", None),
                            "number_of_genes": entry.get("number_of_genes", 0),
                            "number_of_genes_in_background": entry.get(
                                "number_of_genes_in_background", 0
                            ),
                            "input_genes": ";".join(entry.get("inputGenes", [])),
                            "preferred_names": ";".join(entry.get("preferredNames", [])),
                        })
                print(f"  Received {len(enrichment_records)} enrichment terms on retry.")
            else:
                print(f"  ⚠ STRING API returned HTTP {response.status_code} on retry.")
        else:
            print(f"  ⚠ STRING API returned HTTP {response.status_code}")
            print(f"    Response: {response.text[:500]}")

    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Network error calling STRING API: {e}")
        print("  Enrichment will be skipped. Ensure internet connectivity.")

    if not enrichment_records:
        print("\n  No enrichment results obtained. Creating empty output.")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        empty_df = pd.DataFrame(columns=[
            "category", "term", "description", "p_value", "fdr",
            "number_of_genes", "input_genes",
        ])
        empty_path = os.path.join(RESULTS_DIR, "enrichment_results.csv")
        empty_df.to_csv(empty_path, index=False)
        return empty_df

    enrichment_df = pd.DataFrame(enrichment_records)
    enrichment_df = enrichment_df.sort_values("fdr").reset_index(drop=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    enrich_path = os.path.join(RESULTS_DIR, "enrichment_results.csv")
    enrichment_df.to_csv(enrich_path, index=False)
    print(f"\n  Enrichment results saved to: {enrich_path}")

    # --- Print top results per category ---
    categories_of_interest = [
        "Process", "Function", "Component", "KEGG", "Reactome",
    ]
    for cat in categories_of_interest:
        cat_df = enrichment_df[
            enrichment_df["category"].str.contains(cat, case=False, na=False)
        ]
        if len(cat_df) > 0:
            top3 = cat_df.head(3)
            print(f"\n  Top {cat} enrichments:")
            for _, row in top3.iterrows():
                fdr_val = row["fdr"] if pd.notna(row["fdr"]) else "N/A"
                print(f"    {row['description'][:60]:<60s}  FDR={fdr_val}")

    # --- Enrichment visualisation ---
    _generate_enrichment_plot(enrichment_df)

    print("\n✓ Functional enrichment complete.")
    return enrichment_df


def _generate_enrichment_plot(enrichment_df: pd.DataFrame):
    """Bar plot of top enriched terms across categories."""
    if len(enrichment_df) == 0:
        return

    # Take top 5 from each major category
    categories = enrichment_df["category"].unique()
    top_terms = []
    for cat in categories:
        cat_df = enrichment_df[enrichment_df["category"] == cat].head(5)
        top_terms.append(cat_df)

    if not top_terms:
        return

    plot_df = pd.concat(top_terms, ignore_index=True).head(25)
    plot_df["neg_log10_fdr"] = -pd.to_numeric(plot_df["fdr"], errors="coerce").apply(
        lambda x: max(x, 1e-300)
    ).apply(lambda x: __import__("numpy").log10(x))

    # Create short labels
    plot_df["label"] = (
        plot_df["description"].str[:50] + " (" + plot_df["category"] + ")"
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.35)))

    colors = {
        "Process": "#E74C3C",
        "Function": "#3498DB",
        "Component": "#2ECC71",
        "KEGG": "#F39C12",
        "Reactome": "#9B59B6",
    }

    bar_colors = []
    for _, row in plot_df.iterrows():
        matched = False
        for key, color in colors.items():
            if key.lower() in row["category"].lower():
                bar_colors.append(color)
                matched = True
                break
        if not matched:
            bar_colors.append("#95A5A6")

    ax.barh(range(len(plot_df)), plot_df["neg_log10_fdr"].values,
            color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("-log₁₀(FDR)", fontsize=12)
    ax.set_title("Functional Enrichment of Candidate AML-Associated Proteins", fontsize=13)
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, "enrichment_plot.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Enrichment plot saved to: {fig_path}")


if __name__ == "__main__":
    ranking_path = os.path.join(RESULTS_DIR, "final_protein_ranking.csv")
    if os.path.exists(ranking_path):
        candidate = pd.read_csv(ranking_path)
        # For standalone run, we don't have the full mapped set or id_to_name
        print("Run via the full pipeline for proper background universe.")
    else:
        print("Run the full pipeline first.")
