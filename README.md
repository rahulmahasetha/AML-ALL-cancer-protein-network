# AML/ALL PPI + GNN Pipeline

## Overview

This project implements a three-stage pipeline for identifying and prioritizing biomarker proteins for Acute Myeloid Leukemia (AML) and Acute Lymphoblastic Leukemia (ALL).

1. **Stage 1 — Discovery**: Performs two-class differential gene expression analysis between AML and ALL, constructs a unified Protein-Protein Interaction (PPI) network, and computes network centrality to rank candidate proteins.
2. **Stage 2 — Prediction**: Trains a Graph Neural Network (GNN) on patient-specific protein graphs to classify AML vs. ALL. Evaluates models and extracts GNN importance scores for proteins.
3. **Stage 3 — Interpretation**: Compares the top candidate proteins identified by network centrality in Stage 1 with the top important proteins identified by the GNN in Stage 2 to discover highly robust biomarkers.

## Pipeline Workflow

### Stage 1: Discovery
- Compute differential expression (AML-enriched vs. ALL-enriched).
- Map differentially expressed genes to STRING proteins.
- Build a unified PPI network (seed proteins + 1-hop neighbors).
- Compute node centralities (degree, betweenness, closeness).
- Permutation tests & Functional Enrichment for AML, ALL, and Combined sets.

### Stage 2: Prediction
- Construct patient-specific protein graphs using expression features.
- Train GCN/GAT/GraphSAGE to classify AML vs ALL.
- Extract protein importance scores (e.g. via integrated gradients or permutation).

### Stage 3: Interpretation
- Compare Stage 1 candidates with Stage 2 GNN-important proteins.
- Compute overlap for AML-enriched, ALL-enriched, and Combined protein sets.
- Generate overlap visualizations and final interpretation summary.

## Output Files

| File/Directory | Description |
|----------------|-------------|
| `outputs/experiment_summary.json` | High-level metrics for the entire run |
| `outputs/stage1/aml_enriched_proteins.csv` | AML-enriched mapping |
| `outputs/stage1/all_enriched_proteins.csv` | ALL-enriched mapping |
| `outputs/stage1/differential_proteins.csv` | Combined mapping |
| `outputs/stage1/unified_ppi_network_edges.csv` | Edge list for unified PPI network |
| `outputs/stage1/ppi_node_metadata.csv` | Node list with AML/ALL/Neighbor labels |
| `outputs/stage1/protein_network_ranking.csv` | Centrality rankings for combined seeds |
| `outputs/stage1/aml_protein_ranking.csv` | Centrality rankings for AML seeds |
| `outputs/stage1/all_protein_ranking.csv` | Centrality rankings for ALL seeds |
| `outputs/stage1/enrichment/` | Separated functional enrichment results |
| `outputs/stage3/aml_gnn_overlap.csv` | AML-enriched proteins overlapping with GNN importance |
| `outputs/stage3/all_gnn_overlap.csv` | ALL-enriched proteins overlapping with GNN importance |
| `outputs/stage3/combined_gnn_overlap.csv` | Combined overlap |
| `outputs/stage3/interpretation_summary.csv` | Final interpretation metrics |

## Quick Start

```bash
# Run Stage 1 (Discovery)
python3 stage1/code/run_pipeline.py

# Run Stage 2 (Prediction)
python3 stage2/code/04_train_gnn.py
python3 stage2/code/05_evaluate_model.py

# Run Stage 3 (Interpretation)
python3 stage_3_interpretation/code/interpretation.py
```
