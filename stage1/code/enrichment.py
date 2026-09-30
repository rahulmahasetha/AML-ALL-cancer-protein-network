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
        The ranked AML seed proteins (from protein_network_ranking.csv).
        Must have column 'protein' with STRING IDs.
    all_mapped_proteins : set[str]
        All successfully mapped STRING protein IDs (background universe).
    id_to_name : dict[str, str]
        STRING ID → preferred_name.

    Returns
    -------
    enrichment_results : dict
        Dict containing enrichment DataFrames for 'AML', 'ALL', 'COMBINED'.
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

    aml_candidates = candidate_proteins[candidate_proteins["association_group"] == "AML-enriched"]
    all_candidates = candidate_proteins[candidate_proteins["association_group"] == "ALL-enriched"]

    def perform_group_enrichment(proteins_df, group_name):
        top_n = min(TOP_N_PROTEINS, len(proteins_df))
        if top_n == 0:
            return pd.DataFrame(columns=["category", "term", "description", "p_value", "fdr", "number_of_genes", "input_genes"])
        
        query_proteins = proteins_df.head(top_n)["protein_id"].tolist()
        query_identifiers = [id_to_name.get(pid, pid.replace("9606.", "")) for pid in query_proteins]
        background_identifiers = [id_to_name.get(pid, pid.replace("9606.", "")) for pid in all_mapped_proteins]
        
        print(f"\n  --- Enrichment for {group_name} ---")
        print(f"  Query proteins: {len(query_identifiers)}")
        
        identifier_str = "%0d".join(query_identifiers)
        background_str = "%0d".join(background_identifiers)
        
        params_with_bg = {
            "identifiers": identifier_str,
            "species": STRING_SPECIES,
            "background_string_identifiers": background_str,
            "caller_identity": f"aml_ppi_stage1_analysis_{group_name}",
        }
        params_no_bg = {
            "identifiers": identifier_str,
            "species": STRING_SPECIES,
            "caller_identity": f"aml_ppi_stage1_analysis_{group_name}",
        }
        
        enrichment_records = []
        try:
            response = requests.post(STRING_ENRICHMENT_URL, data=params_with_bg, timeout=120)
            results = []
            if response.status_code == 200:
                results = response.json() if isinstance(response.json(), list) else []
            
            if len(results) <= 1:
                response = requests.post(STRING_ENRICHMENT_URL, data=params_no_bg, timeout=120)
                if response.status_code == 200:
                    results = response.json() if isinstance(response.json(), list) else []
            
            if response.status_code == 200 and isinstance(results, list):
                for entry in results:
                    enrichment_records.append({
                        "category": entry.get("category", ""),
                        "term": entry.get("term", ""),
                        "description": entry.get("description", ""),
                        "p_value": entry.get("p_value", None),
                        "fdr": entry.get("fdr", None),
                        "number_of_genes": entry.get("number_of_genes", 0),
                        "number_of_genes_in_background": entry.get("number_of_genes_in_background", 0),
                        "input_genes": ";".join(entry.get("inputGenes", [])),
                        "preferred_names": ";".join(entry.get("preferredNames", [])),
                    })
        except Exception as e:
            print(f"  ⚠ Network error calling STRING API for {group_name}: {e}")
            
        if not enrichment_records:
            return pd.DataFrame(columns=["category", "term", "description", "p_value", "fdr", "number_of_genes", "input_genes"])
            
        enrichment_df = pd.DataFrame(enrichment_records)
        return enrichment_df.sort_values("fdr").reset_index(drop=True)

    results_dict = {}

    for group_name, df_group in [("AML", aml_candidates), ("ALL", all_candidates), ("COMBINED", candidate_proteins)]:
        enrichment_df = perform_group_enrichment(df_group, group_name)
        results_dict[group_name] = enrichment_df

        group_dir = os.path.join(RESULTS_DIR, "enrichment", group_name)
        os.makedirs(group_dir, exist_ok=True)

        enrich_path = os.path.join(group_dir, "enrichment_results.csv")
        enrichment_df.to_csv(enrich_path, index=False)
        print(f"  Saved {group_name} enrichment to {enrich_path}")

        if len(enrichment_df) > 0:
            _generate_enrichment_plot(enrichment_df, group_name, group_dir)

    # --- Consolidated summary: AML + ALL side by side in one file ---
    aml_df = results_dict.get("AML", pd.DataFrame())
    all_df = results_dict.get("ALL", pd.DataFrame())

    frames = []
    if len(aml_df) > 0:
        aml_df = aml_df.copy()
        aml_df.insert(0, "group", "AML-enriched")
        frames.append(aml_df)
    if len(all_df) > 0:
        all_df = all_df.copy()
        all_df.insert(0, "group", "ALL-enriched")
        frames.append(all_df)

    if frames:
        consolidated = pd.concat(frames, ignore_index=True).sort_values(["group", "fdr"])
        consolidated_path = os.path.join(RESULTS_DIR, "enrichment_results_all_groups.csv")
        consolidated.to_csv(consolidated_path, index=False)
        print(f"\n  Consolidated AML + ALL enrichment saved to: {consolidated_path}")

    # Copy AML and ALL plots to the top-level stage1 folder for easy access
    import shutil
    for group_name in ("AML", "ALL"):
        src = os.path.join(RESULTS_DIR, "enrichment", group_name, "enrichment_plot.png")
        dst = os.path.join(RESULTS_DIR, f"enrichment_plot_{group_name.lower()}.png")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  {group_name} enrichment plot copied to: {dst}")

    print("\n✓ Functional enrichment complete.")
    return results_dict


def _generate_enrichment_plot(enrichment_df: pd.DataFrame, group_name: str, group_dir: str):
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
    ax.set_title(f"Functional Enrichment of {group_name} Candidate Proteins", fontsize=13)
    plt.tight_layout()

    fig_path = os.path.join(group_dir, "enrichment_plot.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Enrichment plot saved to: {fig_path}")


if __name__ == "__main__":
    ranking_path = os.path.join(RESULTS_DIR, "protein_network_ranking.csv")
    if os.path.exists(ranking_path):
        candidate = pd.read_csv(ranking_path)
        # For standalone run, we don't have the full mapped set or id_to_name
        print("Run via the full pipeline for proper background universe.")
    else:
        print("Run the full pipeline first.")
