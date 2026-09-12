# Stage 1: Candidate AML-Associated Protein Identification via PPI Network Analysis

## Overview

This pipeline identifies and ranks **candidate AML (Acute Myeloid Leukemia)-associated proteins** by integrating gene-expression analysis with protein–protein interaction (PPI) network topology. It is a complete, reproducible traditional bioinformatics analysis — no deep learning or GNN is used at this stage.

> **Important**: All outputs are *candidate* AML-associated proteins — proteins statistically associated with AML gene-expression signatures that occupy central positions in the PPI network. They are not experimentally validated AML drivers.

## Pipeline Workflow

```
Golub AML/ALL gene-expression dataset
        ↓
1. Load & clean expression data, verify labels from actual.csv
        ↓
2. AML vs ALL differential expression (Welch's t-test, BH-FDR)
        ↓
3. Select AML-relevant genes (adj. p < 0.05, |log2FC| > 1.0)
        ↓
4. Gene → STRING protein mapping (reliable extraction, no guessing)
        ↓
5. Load STRING PPI (confidence ≥ 700), build AML-seed + 1-hop network
        ↓
6. Build NetworkX graph with seed/neighbor node tagging
        ↓
7. Compute centrality: degree, betweenness, closeness
        ↓
8. Combined ranking (configurable weights); seed-only final candidates
        ↓
9. Permutation test: p = (count + 1) / (N + 1)
        ↓
10. GO/pathway enrichment (STRING API, mapped proteins as background)
        ↓
11. Independent dataset validation (no data leakage)
        ↓
Final candidate AML-associated proteins
```

## Input Data

| File | Description |
|------|-------------|
| `data_set_ALL_AML_train.csv` | Golub et al. gene-expression data (7,129 genes × 38 samples) |
| `actual.csv` | Patient labels (AML/ALL), used for both training and independent sample identification |
| `data_set_ALL_AML_independent.csv` | Independent validation set (used ONLY for final validation) |
| `9606.protein.info.v12.0.txt` | STRING v12.0 human protein metadata |
| `9606.protein.links.v12.0.txt` | STRING v12.0 human PPI interactions (~13.7M edges) |

## Configuration

All thresholds are configurable in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `ADJUSTED_PVALUE_THRESHOLD` | 0.05 | BH-adjusted p-value cutoff |
| `LOG2FC_THRESHOLD` | 1.0 | Minimum |log2 fold-change| |
| `STRING_CONFIDENCE_THRESHOLD` | 700 | Minimum STRING combined_score |
| `CENTRALITY_WEIGHTS` | {degree: 0.4, betweenness: 0.3, closeness: 0.3} | Weighted combination for ranking |
| `DISTANCE_EPSILON` | 1e-6 | Epsilon for distance = 1/(score/1000 + ε) |
| `N_PERMUTATIONS` | 1000 | Number of permutation iterations |
| `TOP_N_PROTEINS` | 50 | Number of top proteins to report |

## Key Methodological Choices

### Differential Expression
- **Test**: Welch's t-test (unequal variance) per gene
- **Multiple testing**: Benjamini-Hochberg FDR correction
- **Effect size**: log2 fold-change with pseudocount (ε=1) to avoid log(0)

### Gene → Protein Mapping
- Gene symbols are extracted from Affymetrix probe descriptions using curated patterns
- Matched against STRING `preferred_name` using exact case-insensitive matching
- **Unmapped genes are left unmapped** — no heuristic guessing
- Mapping percentage is computed at runtime

### PPI Network
- Edges filtered by configurable STRING confidence threshold
- Network includes AML seed proteins and their 1-hop PPI neighbours
- Seed vs neighbour nodes are tagged and reported separately

### Weighted Centrality
- STRING combined_score converted to distance: `distance = 1 / (score/1000 + ε)`
- This ensures higher-confidence interactions correspond to shorter distances
- All distances are strictly positive

### Ranking
- **Two rankings** are generated:
  1. All-network proteins (for context)
  2. **AML seed proteins only** (used as final candidate list)
- Centrality weights are experimental choices, not biological constants

### Permutation Test
- Tests whether AML seed mean centrality exceeds chance expectation
- Corrected p-value: `(count(null ≥ observed) + 1) / (N + 1)`

### Enrichment
- Background universe: all successfully mapped proteins (not full human proteome)
- Uses STRING API for GO (BP/MF/CC), KEGG, Reactome

### Data Leakage Prevention
- The independent dataset is used **only** for final validation
- No gene selection, threshold tuning, or network construction uses independent data

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python3 src/run_pipeline.py
```

## Output Files

| File | Description |
|------|-------------|
| `results/differential_expression.csv` | Full DE results for all genes |
| `results/selected_aml_genes.csv` | Genes passing selection criteria |
| `results/volcano_plot.png` | Volcano plot of DE results |
| `results/gene_protein_mapping.csv` | Gene → STRING protein mapping |
| `results/ppi_edge_table.csv` | PPI network edge list |
| `results/ppi_node_table.csv` | PPI node list with seed/neighbor tags |
| `results/network_statistics.csv` | Network summary statistics |
| `results/ppi_network.png` | Network visualisation |
| `results/protein_centrality_all.csv` | Centrality for all network proteins |
| `results/final_protein_ranking.csv` | Ranked AML seed proteins (final candidates) |
| `results/top_proteins_ranking.png` | Bar chart of top candidates |
| `results/permutation_test.csv` | Permutation test results |
| `results/permutation_null_distribution.png` | Null distribution histogram |
| `results/enrichment_results.csv` | GO/pathway enrichment results |
| `results/enrichment_plot.png` | Enrichment visualisation |
| `results/independent_validation.csv` | Independent dataset validation |

## Project Structure

```
aml_ppi_stage1/
├── data/                           # Input data (symlinks)
├── src/
│   ├── config.py                   # All configurable parameters
│   ├── data_loader.py              # Step 1: Data loading & cleaning
│   ├── differential_expression.py  # Steps 2-3: DE analysis & gene selection
│   ├── gene_protein_mapping.py     # Step 4: Gene → STRING mapping
│   ├── ppi_network.py              # Steps 5-6: PPI network construction
│   ├── network_analysis.py         # Steps 7-8: Centrality & ranking
│   ├── permutation_test.py         # Step 9: Permutation validation
│   ├── enrichment.py               # Step 10: Functional enrichment
│   ├── validation.py               # Step 11: Independent validation
│   └── run_pipeline.py             # Master orchestrator
├── results/                        # All outputs
├── requirements.txt
└── README.md
```

## Biological Interpretation Note

Known AML genes such as FLT3, KIT, or NPM1 may appear in the results and can be used for qualitative biological interpretation — their presence suggests the pipeline produces biologically plausible results. However, their appearance is **not** a required success criterion. The pipeline's statistical validity rests on the permutation test and enrichment analysis.

## Dependencies

- Python ≥ 3.9
- pandas, numpy, scipy, statsmodels, networkx, matplotlib, seaborn, scikit-learn, requests
