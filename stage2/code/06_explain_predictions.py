import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import importlib.util
import os
from torch_geometric.loader import DataLoader

# Load the GNNClassifier from the training script
spec = importlib.util.spec_from_file_location("train_gnn", "stage2/code/04_train_gnn.py")
train_gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_gnn)
GNNClassifier = train_gnn.GNNClassifier

def main():
    print("Computing GNN feature importances via Input x Gradients...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_graphs = torch.load('stage2/results/train_graphs.pt', weights_only=False)
    loader = DataLoader(train_graphs, batch_size=1, shuffle=False)
    
    # Load model
    save_dict = torch.load('stage2/results/best_gnn_ensemble.pt', map_location=device, weights_only=True)
    model_type = save_dict['model_type']
    params = save_dict['params']
    model_states = save_dict['model_states']
    
    model = GNNClassifier(
        num_node_features=1, 
        hidden_channels=params['hidden_dim'], 
        num_classes=2,
        model_type=model_type,
        dropout=params['dropout'],
        pooling=params['pooling']
    )
    model.to(device)
    
    # We will average importances over all models in the ensemble
    num_nodes = train_graphs[0].x.size(0)
    total_importances = np.zeros(num_nodes)
    
    # Also load the node names from stage 1 metadata
    nodes_df = pd.read_csv("outputs/stage1/ppi_node_metadata.csv")
    
    for state in model_states:
        model.load_state_dict(state)
        model.eval()
        
        for data in loader:
            data = data.to(device)
            data.x.requires_grad_(True)
            
            out = model(data.x, data.edge_index, data.batch)
            # Use the score of the true class
            score = out[0, data.y.item()]
            
            model.zero_grad()
            score.backward()
            
            grad = data.x.grad.data.cpu().numpy().squeeze()
            x_val = data.x.data.cpu().numpy().squeeze()
            
            # Input x Gradient
            importance = np.abs(grad * x_val)
            total_importances += importance
            
    # Average across all ensemble models and all patients
    avg_importances = total_importances / (len(model_states) * len(train_graphs))
    
    nodes_df["importance_score"] = avg_importances
    
    # Sort by importance
    nodes_df = nodes_df.sort_values(by="importance_score", ascending=False)
    
    out_file = "stage2/results/important_proteins.csv"
    nodes_df.to_csv(out_file, index=False)
    
    print(f"Top 10 most important proteins:")
    print(nodes_df.head(10)[["protein_id", "protein_name", "importance_score"]])
    print(f"Importances saved to {out_file}")

if __name__ == "__main__":
    main()
