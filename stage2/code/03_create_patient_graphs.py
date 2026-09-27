import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    stage1_out = os.path.join(base_dir, "outputs", "stage1")
    
    # 1. Labels
    labels_df = pd.read_csv(os.path.join(data_dir, "actual.csv"))
    # The file has format: patient,cancer.  Where cancer is 'AML' or 'ALL'
    patient_to_label = {}
    for _, row in labels_df.iterrows():
        # Clean patient string e.g., '1' or '2'
        pid = str(row['patient']).strip()
        label = 1 if row['cancer'].strip().upper() == 'AML' else 0
        patient_to_label[pid] = label
        
    # 2. Expression Data
    train_exp = pd.read_csv(os.path.join(data_dir, "data_set_ALL_AML_train.csv"))
    test_exp = pd.read_csv(os.path.join(data_dir, "data_set_ALL_AML_independent.csv"))
    
    # Process expression columns: drop 'Gene Description', 'Gene Accession' (or 'Gene Accession Number')
    # and just keep patient ID columns (integers usually).
    # Wait, the structure usually has 'Gene Accession', 'Gene Description' and then patient columns + 'call' columns.
    
    def process_expression(exp_df):
        cols = exp_df.columns
        patient_cols = [c for c in cols if str(c).isdigit()]
        
        # Determine the gene column: 'Gene Accession' or 'Gene Accession Number'
        gene_col = 'Gene Accession'
        if gene_col not in cols:
            gene_col = 'Gene Accession Number'
        if gene_col not in cols:
            # Fallback to whatever looks like gene accession
            gene_col = cols[1] if len(cols) > 1 else cols[0]
            
        data_dict = {}
        for pid in patient_cols:
            data_dict[pid] = pd.Series(exp_df[pid].values, index=exp_df[gene_col].values)
        return data_dict
        
    train_patients = process_expression(train_exp)
    test_patients = process_expression(test_exp)
    
    # 3. Gene to Protein Mapping
    mapping_df = pd.read_csv(os.path.join(stage1_out, "differential_proteins.csv"))
    protein_to_genes = {}
    for _, row in mapping_df.iterrows():
        prot = row["protein_id"]
        gene = row["gene"]
        if prot not in protein_to_genes:
            protein_to_genes[prot] = []
        protein_to_genes[prot].append(gene)
        
    # 4. Nodes and Edges
    nodes_df = pd.read_csv(os.path.join(stage1_out, "ppi_node_metadata.csv"))
    edges_df = pd.read_csv(os.path.join(stage1_out, "unified_ppi_network_edges.csv"))
    
    return patient_to_label, train_patients, test_patients, protein_to_genes, nodes_df, edges_df

def build_graphs(patient_data_dict, patient_to_label, protein_to_genes, nodes_df, edges_df):
    graphs = []
    
    # Create mapping from protein_id to node_index (0-based)
    protein_to_idx = {row["protein_id"]: idx for idx, row in nodes_df.iterrows()}
    
    # Create edge_index
    source_nodes = []
    target_nodes = []
    for _, row in edges_df.iterrows():
        p1 = row["protein_A"]
        p2 = row["protein_B"]
        if p1 in protein_to_idx and p2 in protein_to_idx:
            source_nodes.append(protein_to_idx[p1])
            target_nodes.append(protein_to_idx[p2])
            # undirected:
            source_nodes.append(protein_to_idx[p2])
            target_nodes.append(protein_to_idx[p1])
            
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    num_nodes = len(nodes_df)
    
    for pid, exp_series in patient_data_dict.items():
        if pid not in patient_to_label:
            continue
            
        y = torch.tensor([patient_to_label[pid]], dtype=torch.long)
        
        # Features
        x = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        for idx, row in nodes_df.iterrows():
            prot = row["protein_id"]
            if prot in protein_to_genes:
                genes = protein_to_genes[prot]
                # Average expression of genes that map to this protein
                vals = []
                for g in genes:
                    if g in exp_series:
                        vals.append(exp_series[g])
                if vals:
                    x[idx, 0] = np.mean(vals)
                    
        # Apply z-score normalization to the non-zero features per patient?
        # Standard GNN practice is feature normalization. Let's do z-score across all non-zero nodes for the patient
        non_zeros = x[x != 0]
        if len(non_zeros) > 0:
            mean = non_zeros.mean()
            std = non_zeros.std()
            if std > 0:
                x[x != 0] = (x[x != 0] - mean) / std
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.patient_id = pid
        graphs.append(data)
        
    return graphs

def main():
    print("Loading data for graph construction...")
    patient_to_label, train_patients, test_patients, protein_to_genes, nodes_df, edges_df = load_data()
    
    print("Building training graphs...")
    train_graphs = build_graphs(train_patients, patient_to_label, protein_to_genes, nodes_df, edges_df)
    
    print("Building testing graphs...")
    test_graphs = build_graphs(test_patients, patient_to_label, protein_to_genes, nodes_df, edges_df)
    
    # Split train into train/val (e.g. 80/20)
    # The dataset has 38 train samples. 80% is 30.
    np.random.seed(42)
    indices = np.random.permutation(len(train_graphs))
    split = int(0.8 * len(train_graphs))
    train_split = [train_graphs[i] for i in indices[:split]]
    val_split = [train_graphs[i] for i in indices[split:]]
    
    print(f"Generated {len(train_split)} train graphs, {len(val_split)} val graphs, and {len(test_graphs)} test graphs.")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save(train_split, os.path.join(out_dir, "train_graphs.pt"))
    torch.save(val_split, os.path.join(out_dir, "val_graphs.pt"))
    torch.save(test_graphs, os.path.join(out_dir, "test_graphs.pt"))
    print(f"Graphs saved to {out_dir}")

if __name__ == "__main__":
    main()
