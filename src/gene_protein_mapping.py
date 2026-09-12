"""
Step 4: Gene → STRING Protein Mapping.

Maps selected AML-relevant genes to STRING protein IDs using the
9606.protein.info.v12.0.txt reference. Uses reliable extraction of
gene symbols from Affymetrix probe descriptions — genes that cannot
be reliably mapped are left unmapped (no guessing).
"""

import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STRING_INFO_FILE, RESULTS_DIR


# Common English words that should NOT be treated as gene symbols
_STOPWORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "NOT", "ARE",
    "WAS", "BUT", "HAS", "HIS", "HER", "ALL", "CAN", "HAD", "ONE",
    "OUR", "NEW", "NOW", "OLD", "SEE", "WAY", "MAY", "DAY", "TOO",
    "ANY", "WHO", "BOY", "DID", "GET", "LET", "SAY", "SHE", "USE",
    "MAN", "BIG", "END", "PUT", "RUN", "SET", "TRY", "ASK", "MEN",
    "GENE", "HUMAN", "PROTEIN", "MRNA", "CDNA", "CLONE", "PARTIAL",
    "CDS", "DEF", "COMPLETE", "SEQUENCE", "ENCODING", "SUBUNIT",
    "RECEPTOR", "KINASE", "FACTOR", "BINDING", "DOMAIN", "FAMILY",
    "MEMBER", "TYPE", "ISOFORM", "ALPHA", "BETA", "GAMMA", "DELTA",
    "LIKE", "RELATED", "ASSOCIATED", "CONTAINING", "PRECURSOR",
    "CHAIN", "LARGE", "SMALL", "LONG", "SHORT", "LIGHT", "HEAVY",
    "SIMILAR", "HOMOLOG", "VARIANT", "FRAGMENT", "FULL", "LENGTH",
    "EXON", "INTRON", "ANTI", "PRO", "PRE", "CELL", "CELLS",
    "TERMINAL", "CYTOPLASMIC", "NUCLEAR", "MEMBRANE", "SURFACE",
    "MUSCLE", "LIVER", "BRAIN", "HEART", "LUNG", "BONE", "BLOOD",
    "ACID", "BASE", "MOTIF", "REPEAT", "RICH", "DEPENDENT",
    "INDEPENDENT", "SPECIFIC", "ACTIVATED", "INDUCED", "INHIBITOR",
    "ACTIVATOR", "REGULATOR", "SUPPRESSOR", "TRANSPORTER",
    "CHANNEL", "PUMP", "SYNTHASE", "TRANSFERASE", "LYASE", "LIGASE",
    "OXIDASE", "REDUCTASE", "HYDROLASE", "PHOSPHATASE", "PROTEASE",
    "DEHYDROGENASE", "ISOMERASE", "POLYMERASE", "HELICASE", "DNA", "RNA",
}


def load_string_protein_info(info_path: str) -> dict:
    """
    Load STRING protein info and build a lookup:
    preferred_name (uppercase) → string_protein_id.

    Returns
    -------
    name_to_id : dict[str, str]
        Upper-case preferred_name → full STRING ID (e.g. '9606.ENSP00000000233')
    id_to_name : dict[str, str]
        STRING ID → preferred_name (original case)
    """
    df = pd.read_csv(info_path, sep="\t", usecols=["#string_protein_id", "preferred_name"])
    df.columns = ["string_protein_id", "preferred_name"]

    name_to_id = {}
    id_to_name = {}
    for _, row in df.iterrows():
        name_upper = str(row["preferred_name"]).strip().upper()
        sid = str(row["string_protein_id"]).strip()
        name_to_id[name_upper] = sid
        id_to_name[sid] = row["preferred_name"]

    return name_to_id, id_to_name


def extract_gene_symbol(description: str) -> str | None:
    """
    Attempt to extract a reliable gene symbol from an Affymetrix probe
    description string.

    Strategy (applied in order):
    1. If description starts with a token that looks like a standard gene
       symbol (2-15 alphanumeric chars, at least one uppercase letter,
       may include digits and hyphens), extract it.
    2. Handle "GB DEF = ..." prefix: strip it and retry.
    3. Reject tokens that are common English or biology stopwords.
    4. Return None if no reliable symbol can be extracted.
    """
    if not isinstance(description, str) or not description.strip():
        return None

    desc = description.strip()

    # Strip "GB DEF = " prefix
    if desc.upper().startswith("GB DEF"):
        desc = re.sub(r"^GB\s+DEF\s*=\s*", "", desc, flags=re.IGNORECASE).strip()

    # Try to get the first token
    tokens = re.split(r"[\s,;:()/]+", desc)
    if not tokens:
        return None

    for token in tokens[:3]:  # Check first three tokens only
        token = token.strip().strip("'\"")
        if not token:
            continue

        # Must be 2-15 characters, alphanumeric (plus hyphens allowed)
        if not re.match(r"^[A-Za-z][A-Za-z0-9\-]{1,14}$", token):
            continue

        # Must contain at least one uppercase letter
        if not any(c.isupper() for c in token):
            continue

        upper_token = token.upper()

        # Skip stopwords
        if upper_token in _STOPWORDS:
            continue

        # Skip if it looks like a GenBank accession (letter(s) followed by 5-6 digits)
        if re.match(r"^[A-Z]{1,3}\d{5,6}$", upper_token):
            continue

        # Skip AFFX control probes
        if upper_token.startswith("AFFX"):
            continue

        return upper_token

    return None


def map_genes_to_proteins(selected_genes: pd.DataFrame):
    """
    Map selected genes to STRING protein IDs.

    Parameters
    ----------
    selected_genes : pd.DataFrame
        Must contain columns: gene, gene_description.

    Returns
    -------
    mapping_df : pd.DataFrame
        One row per selected gene with mapping result.
    mapped_proteins : set[str]
        Set of successfully mapped STRING protein IDs.
    """
    print("\n" + "=" * 60)
    print("STEP 4: GENE → STRING PROTEIN MAPPING")
    print("=" * 60)

    # Load STRING reference
    print(f"\n  Loading STRING protein info from: {STRING_INFO_FILE}")
    name_to_id, id_to_name = load_string_protein_info(STRING_INFO_FILE)
    print(f"  STRING proteins loaded: {len(name_to_id)}")

    # Map each selected gene
    records = []
    mapped_proteins = set()

    for _, row in selected_genes.iterrows():
        gene_acc = row["gene"]
        gene_desc = row.get("gene_description", "")

        extracted = extract_gene_symbol(gene_desc)
        string_id = None
        protein_name = None
        status = "unmapped"

        if extracted and extracted in name_to_id:
            string_id = name_to_id[extracted]
            protein_name = id_to_name.get(string_id, extracted)
            status = "mapped"
            mapped_proteins.add(string_id)

        records.append({
            "gene": gene_acc,
            "gene_description": gene_desc,
            "extracted_symbol": extracted,
            "string_protein_id": string_id,
            "protein_name": protein_name,
            "mapping_status": status,
        })

    mapping_df = pd.DataFrame(records)

    # --- Report ---
    n_total = len(mapping_df)
    n_mapped = (mapping_df["mapping_status"] == "mapped").sum()
    n_unmapped = n_total - n_mapped
    mapping_pct = round(100 * n_mapped / n_total, 1) if n_total > 0 else 0.0

    print(f"\n  Gene-to-protein mapping results:")
    print(f"    Total selected genes:      {n_total}")
    print(f"    Successfully mapped:       {n_mapped}")
    print(f"    Unmapped:                  {n_unmapped}")
    print(f"    Mapping percentage:        {mapping_pct}%")
    print(f"    Unique STRING proteins:    {len(mapped_proteins)}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    map_path = os.path.join(RESULTS_DIR, "gene_protein_mapping.csv")
    mapping_df.to_csv(map_path, index=False)
    print(f"\n  Mapping table saved to: {map_path}")

    if n_mapped > 0:
        print(f"\n  Sample mapped genes:")
        mapped_rows = mapping_df[mapping_df["mapping_status"] == "mapped"].head(10)
        for _, r in mapped_rows.iterrows():
            print(f"    {r['extracted_symbol']:12s} → {r['string_protein_id']}")

    print("\n✓ Gene-to-protein mapping complete.")
    return mapping_df, mapped_proteins, id_to_name


if __name__ == "__main__":
    from data_loader import load_and_prepare_data
    from differential_expression import run_differential_expression

    data = load_and_prepare_data()
    _, selected = run_differential_expression(
        data["expression"], data["gene_meta"],
        data["aml_sample_ids"], data["all_sample_ids"],
    )
    mapping_df, mapped_proteins, id_to_name = map_genes_to_proteins(selected)
