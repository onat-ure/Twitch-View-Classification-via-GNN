import os
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from models.gnn_regressor import GNNRegressor
import matplotlib.pyplot as plt

device = torch.device('cpu')


def load_model(model_path, num_features):
    model = GNNRegressor(num_features=num_features, hidden_channels=128).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model: {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    return model


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    eps = 1e-8  
    data.y = torch.log(data.y + eps)
    out = model(data.x, data.edge_index)
    predictions = out[data.test_mask]
    true_values = data.y[data.test_mask]
    return predictions, true_values

def calculate_metrics(predictions, true_values):
    # First calculate metrics in log space
    pred_np = predictions.cpu().numpy()
    true_np = true_values.cpu().numpy()
    
    # Add R² calculation in log space
    log_mean_true = torch.mean(true_values)
    log_ss_tot = torch.sum((true_values - log_mean_true) ** 2)
    log_ss_res = torch.sum((true_values - predictions) ** 2)
    log_r2 = 1 - (log_ss_res / log_ss_tot)
    
    # Original space calculations
    eps = 1e-8
    pred_orig = torch.exp(predictions) - eps
    true_orig = torch.exp(true_values) - eps
    
    # Calculate R² in original space with scaling
    orig_mean_true = torch.mean(true_orig)
    
    # Print debug information
    print("\nDebug Information:")
    print(f"Original Space Statistics:")
    print(f"True values - Mean: {orig_mean_true:.2f}, Min: {torch.min(true_orig):.2f}, Max: {torch.max(true_orig):.2f}")
    print(f"Predictions - Mean: {torch.mean(pred_orig):.2f}, Min: {torch.min(pred_orig):.2f}, Max: {torch.max(pred_orig):.2f}")
    
    # Calculate percentage of predictions that are within an order of magnitude
    ratio = pred_orig / true_orig
    ratio_log = pred_np / true_np
    within_10x = (ratio >= 0.1) & (ratio <= 10)
    within_2x = (ratio >= 0.5) & (ratio <= 2)
    within_10x_log = (ratio_log >= 0.1) & (ratio_log <= 10)
    within_2x_log = (ratio_log >= 0.5) & (ratio_log <= 2)
    # Calculate percentage of predictions that are within an order of magnitude
    
    accuracy_10x = (within_10x.sum() / len(ratio)) * 100
    accuracy_2x = (within_2x.sum() / len(ratio)) * 100
    accuracy_10x_log = (within_10x_log.sum() / len(ratio_log)) * 100
    accuracy_2x_log = (within_2x_log.sum() / len(ratio_log  )) * 100
    
    # Calculate R² with better handling of outliers
    orig_ss_tot = torch.sum((true_orig - orig_mean_true) ** 2)
    orig_ss_res = torch.sum((true_orig - pred_orig) ** 2)
    orig_r2 = 1 - (orig_ss_res / orig_ss_tot)
    
    # Calculate relative error
    relative_error = torch.abs(pred_orig - true_orig) / true_orig
    
    metrics = {
        'r2_score_log': log_r2.item(),
        'r2_score_orig': orig_r2.item(),
        'mean_relative_error': relative_error.mean().item(),
        'median_relative_error': relative_error.median().item(),
        'accuracy_within_10x': accuracy_10x.item(),
        'accuracy_within_2x': accuracy_2x.item(),
        'accuracy_within_10x_log': accuracy_10x_log.item(),
        'accuracy_within_2x_log': accuracy_2x_log.item(),
        'mean_pred': torch.mean(pred_orig).item(),
        'std_true': torch.std(true_orig).item(),
        'std_pred': torch.std(pred_orig).item()
    }
    
    return metrics

def plot_predictions(predictions, true_values):
    # Convert from log space to original space
    eps = 1e-8
    pred_orig = torch.exp(predictions).cpu().numpy() - eps
    true_orig = torch.exp(true_values).cpu().numpy() - eps
    
    # Ensure positive values for log scale
    min_positive_true = np.min(true_orig[true_orig > 0])
    min_positive_pred = np.min(pred_orig[pred_orig > 0])
    min_value = min(min_positive_true, min_positive_pred)
    
    # Scatter plot in original space
    plt.figure(figsize=(10, 6))
    plt.scatter(true_orig, pred_orig, alpha=0.5)
    plt.plot([min_value, true_orig.max()], 
             [min_value, true_orig.max()], 
             'r--', label='Perfect Prediction')
    plt.xlabel('True View Count')
    plt.ylabel('Predicted View Count')
    plt.title('Predicted vs Actual View Counts')
    plt.legend()
    plt.grid(True)
    
    # Use log scale for better visualization of wide range
    plt.xscale('log')
    plt.yscale('log')
    
    plt.savefig('prediction_scatter.png')
    plt.close()
    
    # Distribution plot in original space
    plt.figure(figsize=(10, 6))
    
    # Create bins avoiding zero/negative values
    bins = np.logspace(np.log10(min_value), 
                      np.log10(max(true_orig.max(), pred_orig.max())), 
                      50)
    
    # Plot histograms with log-scaled x-axis
    plt.hist(true_orig[true_orig > 0], bins=bins, 
             alpha=0.5, label='True Values')
    plt.hist(pred_orig[pred_orig > 0], bins=bins, 
             alpha=0.5, label='Predictions')
    
    plt.xscale('log')
    plt.xlabel('View Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of True vs Predicted View Counts')
    plt.legend()
    plt.grid(True)
    plt.savefig('distribution_comparison.png')
    plt.close()



def main():
    data_file = "twitch_graph_data.pt"
    data = torch.load(data_file)
    data = data.to(device)
    print(f"Graph data loaded from {data_file}")
    
    
    MODEL_DIR = "saved_models"
    model_number = input("Enter the model number to test (e.g., for best_model5.pt, enter 5): ")
    model_path = os.path.join(MODEL_DIR, f"best_model{model_number}.pt")
    
    try:
        model = load_model(model_path, num_features=data.x.size(1))
        predictions, true_values = evaluate(model, data)
        metrics = calculate_metrics(predictions, true_values)
        
        print("\nTest Results:")
        print(f"Log Space Metrics:")
        print(f"  R² Score (log space): {metrics['r2_score_log']:.4f}")
        print(f"  Predictions within 2x (log space): {metrics['accuracy_within_2x_log']:.1f}%")
        print(f"  Predictions within 10x (log space): {metrics['accuracy_within_10x_log']:.1f}%")
        
        print("\nOriginal Space Metrics:")
        print(f"  R² Score (original): {metrics['r2_score_orig']:.4f}")
        print(f"  Mean Relative Error: {metrics['mean_relative_error']:.4f}")
        print(f"  Median Relative Error: {metrics['median_relative_error']:.4f}")
        print(f"  Predictions within 2x: {metrics['accuracy_within_2x']:.1f}%")
        print(f"  Predictions within 10x: {metrics['accuracy_within_10x']:.1f}%")
        
        plot_predictions(predictions, true_values)
        
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        available_models = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model") and f.endswith(".pt")]
        if available_models:
            print("\nAvailable models:")
            for model in sorted(available_models):
                print(f"- {model}")
        else:
            print("No models found in the saved_models directory.")

if __name__ == "__main__":
    main() 