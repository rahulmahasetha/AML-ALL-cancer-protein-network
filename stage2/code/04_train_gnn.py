import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d
from torch_geometric.loader import DataLoader
import numpy as np
import copy
import os
from sklearn.metrics import roc_auc_score

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, model_type='GCN', dropout=0.5, pooling='mean'):
        super(GNNClassifier, self).__init__()
        self.model_type = model_type
        self.dropout_rate = dropout
        self.pooling = global_mean_pool if pooling == 'mean' else global_max_pool
        
        if model_type == 'GCN':
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(num_node_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.pooling(x, batch)
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin(x)
        return x

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    acc = correct / len(loader.dataset)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5  # Only one class in val set
    return total_loss / len(loader.dataset), acc, roc_auc

def compute_class_weights(dataset):
    labels = [data.y.item() for data in dataset]
    classes = np.unique(labels)
    counts = [labels.count(c) for c in classes]
    total = len(labels)
    # n_samples / (n_classes * np.bincount(y))
    weights = [total / (len(classes) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)

def run_training_seed(model_type, params, train_dataset, val_dataset, seed, device, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = GNNClassifier(
        num_node_features=1, 
        hidden_channels=params['hidden_dim'], 
        num_classes=2,
        model_type=model_type,
        dropout=params['dropout'],
        pooling=params['pooling']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if params.get('use_weights', True) else None)
    
    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, 151):
        for data in train_dataset:
            data.to(device)
        for data in val_dataset:
            data.to(device)
            
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = test(model, val_loader, criterion)
        
        # Early stopping based on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_val_loss, best_val_auc, best_model_state

def main():
    print("Hyperparameter tuning and training GNN models...")
    train_dataset = torch.load('stage2/results/train_graphs.pt', weights_only=False)
    val_dataset = torch.load('stage2/results/val_graphs.pt', weights_only=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    class_weights = compute_class_weights(train_dataset)
    print(f"Class weights (AML, ALL): {class_weights}")
    
    # Define an expanded hyperparameter grid for tuning
    param_grid = [
        {'hidden_dim': 32, 'dropout': 0.3, 'lr': 0.005, 'weight_decay': 5e-4, 'pooling': 'mean', 'use_weights': True},
        {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 1e-4, 'pooling': 'mean', 'use_weights': True},
        {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.005, 'weight_decay': 5e-4, 'pooling': 'max', 'use_weights': True},
        {'hidden_dim': 128, 'dropout': 0.5, 'lr': 0.0005, 'weight_decay': 1e-4, 'pooling': 'mean', 'use_weights': True},
        {'hidden_dim': 64, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 5e-4, 'pooling': 'mean', 'use_weights': False},
    ]
    
    models = ['GCN', 'GAT', 'SAGE']
    seeds = [42, 100, 2023, 777, 999]
    
    best_overall_val_loss = float('inf')
    best_overall_params = None
    best_overall_model_type = None
    best_overall_ensemble_states = []
    
    results_summary = []
    
    for m_type in models:
        print(f"\n--- Tuning {m_type} ---")
        best_grid_val_loss = float('inf')
        best_grid_params = None
        
        # Grid Search on seed 42
        for params in param_grid:
            v_loss, v_auc, _ = run_training_seed(m_type, params, train_dataset, val_dataset, 42, device, class_weights)
            if v_loss < best_grid_val_loss:
                best_grid_val_loss = v_loss
                best_grid_params = params
                
        print(f"Best params for {m_type}: {best_grid_params} (Val Loss: {best_grid_val_loss:.4f})")
        
        # Multiple Seeds Evaluation
        seed_losses = []
        seed_aucs = []
        ensemble_states = []
        
        for s in seeds:
            v_loss, v_auc, m_state = run_training_seed(m_type, best_grid_params, train_dataset, val_dataset, s, device, class_weights)
            seed_losses.append(v_loss)
            seed_aucs.append(v_auc)
            ensemble_states.append(copy.deepcopy(m_state))
            
        mean_loss, std_loss = np.mean(seed_losses), np.std(seed_losses)
        mean_auc, std_auc = np.mean(seed_aucs), np.std(seed_aucs)
        
        print(f"{m_type} across {len(seeds)} seeds -> Val Loss: {mean_loss:.4f} ± {std_loss:.4f}, Val AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        results_summary.append(f"{m_type} | Val Loss: {mean_loss:.4f} ± {std_loss:.4f} | Val AUC: {mean_auc:.4f} ± {std_auc:.4f} | Params: {best_grid_params}")
        
        if mean_loss < best_overall_val_loss:
            best_overall_val_loss = mean_loss
            best_overall_params = best_grid_params
            best_overall_model_type = m_type
            best_overall_ensemble_states = ensemble_states
            
    print(f"\nBest Overall Model: {best_overall_model_type} with Mean Val Loss {best_overall_val_loss:.4f}")
    
    with open('stage2/results/comparison_results.txt', 'w') as f:
        f.write("Model Comparison Results (Mean ± Std over 5 seeds)\n")
        f.write("="*60 + "\n")
        for res in results_summary:
            f.write(res + "\n")
        f.write(f"\nBest Overall Model: {best_overall_model_type}\n")
        f.write(f"Best Params: {best_overall_params}\n")
        
    # Save the architecture parameters along with the ensemble of weights
    save_dict = {
        'model_states': best_overall_ensemble_states,
        'model_type': best_overall_model_type,
        'params': best_overall_params
    }
    torch.save(save_dict, 'stage2/results/best_gnn_ensemble.pt')
    print("Saved best model ensemble and its configuration to stage2/results/best_gnn_ensemble.pt")

if __name__ == "__main__":
    main()