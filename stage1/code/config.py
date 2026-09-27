"""
Configuration for Stage 1: AML PPI Network Analysis Pipeline.

All configurable thresholds and parameters are centralised here.
Modify values as needed for sensitivity analyses or alternative runs.
"""

import os


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "stage1")
STAGE3_RESULTS_DIR = os.path.join(BASE_DIR, "outputs/stage3")
EXPERIMENT_SUMMARY_FILE = os.path.join(BASE_DIR, "outputs/experiment_summary.json")

# Input files
TRAIN_EXPRESSION_FILE = os.path.join(DATA_DIR, "data_set_ALL_AML_train.csv")
LABELS_FILE = os.path.join(DATA_DIR, "actual.csv")
INDEPENDENT_EXPRESSION_FILE = os.path.join(DATA_DIR, "data_set_ALL_AML_independent.csv")
STRING_INFO_FILE = os.path.join(DATA_DIR, "9606.protein.info.v12.0.txt")
STRING_LINKS_FILE = os.path.join(DATA_DIR, "9606.protein.links.v12.0.txt")

# Reproducibility
RANDOM_SEED = 42

# Differential expression thresholds
ADJUSTED_PVALUE_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0            # |log2 fold-change| must exceed this
FOLD_CHANGE_PSEUDOCOUNT = 1.0     # Added to means before log2 to avoid log(0)


# STRING PPI network
STRING_CONFIDENCE_THRESHOLD = 700  # Minimum combined_score (0–1000 scale)


# Centrality & rankin
CENTRALITY_WEIGHTS = {
    "degree": 0.4,
    "betweenness": 0.3,
    "closeness": 0.3,
}
DISTANCE_EPSILON = 1e-6  # For distance = 1 / (score/1000 + epsilon)


# Permutation test
N_PERMUTATIONS = 1000

# Output control
TOP_N_PROTEINS = 50
