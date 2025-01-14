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
    
    pred_np = predictions.cpu().numpy()
    true_np = true_values.cpu().numpy()
    
    mse = torch.nn.MSELoss()(predictions, true_values).item()
    rmse = mse ** 0.5
    mae = mean_absolute_error(true_np, pred_np)
    

    ss_tot = torch.sum((true_values - true_values.mean()) ** 2)
    ss_res = torch.sum((true_values - predictions) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    if ss_tot == 0:
        r2_score = 1.0
    else:
        r2_score = 1 - ss_res / ss_tot

    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score.item(),
    }

def plot_predictions(predictions, true_values):
    
    pred_np = predictions.cpu().numpy()
    true_np = true_values.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(true_np, pred_np, alpha=0.5)
    plt.plot([true_np.min(), true_np.max()], [true_np.min(), true_np.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_scatter.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(true_np, bins=50, alpha=0.5, label='True Values')
    plt.hist(pred_np, bins=50, alpha=0.5, label='Predictions')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of True Values vs Predictions')
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
        print(f"Basic Metrics:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"\nAccuracy Metrics:")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        
        # Plot predictions
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