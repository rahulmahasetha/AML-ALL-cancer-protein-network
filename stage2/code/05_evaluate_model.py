import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import os

# Load the GNNClassifier from the training script
spec = importlib.util.spec_from_file_location("train_gnn", "stage2/code/04_train_gnn.py")
train_gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_gnn)
GNNClassifier = train_gnn.GNNClassifier

def get_ensemble_probabilities(model, model_states, loader, device):
    all_labels = []
    ensemble_probs_class1 = []
    
    for i, state in enumerate(model_states):
        model.load_state_dict(state)
        model.eval()
        
        current_model_probs1 = []
        
        collect_labels = (i == 0)
        
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                probs = F.softmax(out, dim=1)
                
                current_model_probs1.extend(probs[:, 1].cpu().numpy())
                
                if collect_labels:
                    all_labels.extend(data.y.cpu().numpy())
                    
        ensemble_probs_class1.append(current_model_probs1)
        
    mean_probs1 = np.mean(ensemble_probs_class1, axis=0)
    return np.array(all_labels), mean_probs1

def find_optimal_threshold(labels, probs):
    best_thresh = 0.5
    best_f1 = 0.0
    # Search thresholds between 0.1 and 0.9
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def main():
    print("Evaluating ensemble model with Threshold Calibration...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_graphs = torch.load('stage2/results/val_graphs.pt', weights_only=False)
    test_graphs = torch.load('stage2/results/test_graphs.pt', weights_only=False)
    
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=8, shuffle=False)
    
    # Load the best ensemble model details
    save_dict = torch.load('stage2/results/best_gnn_ensemble.pt', map_location=device, weights_only=True)
    model_type = save_dict['model_type']
    params = save_dict['params']
    model_states = save_dict['model_states']
    
    print(f"Loading {model_type} ensemble ({len(model_states)} models) with parameters: {params}")
    
    model = GNNClassifier(
        num_node_features=1, 
        hidden_channels=params['hidden_dim'], 
        num_classes=2,
        model_type=model_type,
        dropout=params['dropout'],
        pooling=params['pooling']
    )
    model.to(device)
    
    print("Extracting probabilities on Validation Set...")
    val_labels, val_probs = get_ensemble_probabilities(model, model_states, val_loader, device)
    
    # Calibrate Threshold
    optimal_threshold = find_optimal_threshold(val_labels, val_probs)
    print(f"Optimal Threshold (maximizes Val F1): {optimal_threshold:.2f}")
    
    print("Extracting probabilities on Test Set...")
    test_labels, test_probs = get_ensemble_probabilities(model, model_states, test_loader, device)
    
    # Final predictions based on calibrated threshold
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(test_labels, test_preds)
    
    try:
        prec = precision_score(test_labels, test_preds, zero_division=0)
        rec = recall_score(test_labels, test_preds, zero_division=0)
        f1 = f1_score(test_labels, test_preds, zero_division=0)
        roc = roc_auc_score(test_labels, test_probs)
    except Exception as e:
        print(f"Warning during metric calculation: {e}")
        prec, rec, f1, roc = 0.0, 0.0, 0.0, 0.0
        
    metrics_df = pd.DataFrame([{
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'ROC-AUC': roc
    }])
    metrics_df.to_csv('stage2/results/stage2_metrics.csv', index=False)
    
    # Save predictions
    preds_df = pd.DataFrame({
        'Actual_Label': test_labels,
        'Predicted_Label': test_preds,
        'Probability_Class1_ALL': test_probs
    })
    preds_df.to_csv('stage2/results/predictions.csv', index=False)
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AML', 'ALL'], yticklabels=['AML', 'ALL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Thresh={optimal_threshold:.2f})')
    plt.savefig('stage2/results/confusion_matrix.png')
    plt.close()
    
    print("\nTest Set Evaluation complete.")
    print(metrics_df)

if __name__ == "__main__":
    main()