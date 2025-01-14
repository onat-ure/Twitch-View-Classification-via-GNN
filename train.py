import os
import torch
from models.gnn_regressor import GNNRegressor
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import NeighborLoader
from utils.plotting import TrainingLogger


device = torch.device("cpu")
print("Using CPU")


MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

files = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model") and f.endswith(".pt")]
model_numbers = [int(f.split("best_model")[1].split(".pt")[0]) for f in files if f.split("best_model")[1].split(".pt")[0].isdigit()]
next_model_number = max(model_numbers, default=0) + 1
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"best_model{next_model_number}.pt")

def save_best_model(model):
    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"\nModel saved as: {BEST_MODEL_PATH}")


data_file = "twitch_graph_data.pt"
data = torch.load(data_file)
print(f"Graph data loaded from {data_file}")

# Print data split information
total_nodes = data.num_nodes
train_nodes = data.train_mask.sum().item()
val_nodes = data.val_mask.sum().item()

print("\nData Split Information:")
print(f"Total nodes: {total_nodes:,}")
print(f"Training nodes: {train_nodes:,} ({train_nodes/total_nodes*100:.1f}%)")
print(f"Validation nodes: {val_nodes:,} ({val_nodes/total_nodes*100:.1f}%)")

# -----------------------------------
# Create Data Loaders with PyG-Lib
# -----------------------------------
BATCH_SIZE = 1024
NUM_NEIGHBORS = [10, 10]  # Number of neighbors to sample for each layer

train_loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    input_nodes=data.train_mask,
    shuffle=True,
)

val_loader = NeighborLoader(
    data,
    num_neighbors=NUM_NEIGHBORS,
    batch_size=BATCH_SIZE,
    input_nodes=data.val_mask,
)

print(f"\nBatch Information:")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of training batches: {len(train_loader):,}")
print(f"Number of validation batches: {len(val_loader):,}")

# -----------------------------------
# Initialize Model and Training Components
# -----------------------------------
model = GNNRegressor(num_features=data.x.size(1), hidden_channels=64).to(device)
initial_lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-5,
    verbose=True
)

# Add gradient clipping
MAX_GRAD_NORM = 1.0

loss_fn = torch.nn.MSELoss()

def calculate_accuracy(pred, target, threshold=0.1):
    abs_diff = torch.abs(pred - target)
    mean_val = torch.mean(torch.abs(target))
    within_threshold = abs_diff <= (threshold * mean_val)
    accuracy = torch.mean(within_threshold.float()) * 100
    return accuracy.item()

def train_epoch():
    model.train()
    total_loss = 0
    total_acc = 0
    nodes_processed = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        
        # Only count unique nodes in the batch
        batch_mask = batch.train_mask
        batch_nodes = len(torch.unique(batch_mask.nonzero()))
        
        loss = loss_fn(out[batch_mask], batch.y[batch_mask])
        loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(out[batch_mask], batch.y[batch_mask])
        
        total_loss += loss.item() * batch_nodes
        total_acc += acc * batch_nodes
        nodes_processed += batch_nodes
        
        del batch, out
    
    # Ensure nodes_processed doesn't exceed total training nodes
    nodes_processed = min(nodes_processed, train_nodes)
    return total_loss / nodes_processed, total_acc / nodes_processed, nodes_processed

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0
    total_acc = 0
    nodes_processed = 0
    
    for batch in val_loader:
        out = model(batch.x, batch.edge_index)
        
        # Get predictions for the input nodes
        batch_mask = batch.val_mask
        loss = loss_fn(out[batch_mask], batch.y[batch_mask])
        acc = calculate_accuracy(out[batch_mask], batch.y[batch_mask])
        batch_nodes = batch_mask.sum().item()
        
        total_loss += loss.item() * batch_nodes
        total_acc += acc * batch_nodes
        nodes_processed += batch_nodes
        
        del batch, out
    
    return total_loss / nodes_processed, total_acc / nodes_processed, nodes_processed

# -----------------------------------
# Training Loop
# -----------------------------------
best_val_loss = float('inf')
patience = 15  # Increased patience
patience_counter = 0
n_epochs = 300  # Increased max epochs

logger = TrainingLogger()

# Log initial setup
logger.log(f"Device: {device}")
logger.log(f"\nData Split Information:")
logger.log(f"Total nodes: {total_nodes:,}")
logger.log(f"Training nodes: {train_nodes:,} ({train_nodes/total_nodes*100:.1f}%)")
logger.log(f"Validation nodes: {val_nodes:,} ({val_nodes/total_nodes*100:.1f}%)")
logger.log(f"\nBatch Information:")
logger.log(f"Batch size: {BATCH_SIZE}")
logger.log(f"Number of training batches: {len(train_loader):,}")
logger.log(f"Number of validation batches: {len(val_loader):,}")

print("\nStarting training...")
logger.log("\nStarting training...")
pbar = tqdm(range(1, n_epochs + 1), desc="Training")
for epoch in pbar:
    train_loss, train_acc, train_nodes = train_epoch()
    val_loss, val_acc, val_nodes = evaluate()
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log metrics
    logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
    
    # Early stopping check with more detailed logging
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_best_model(model)
        logger.log(f"New best model saved with validation loss: {val_loss:.4f}", print_to_console=False)
    else:
        patience_counter += 1
        if patience_counter == patience:
            logger.log(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            logger.log(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Update progress bar with learning rate
    pbar.set_description(
        f"Epoch {epoch:03d} | "
        f"LR: {current_lr:.2e} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"Train Nodes: {train_nodes:,}"
    )

    # Log every 10 epochs
    if epoch % 10 == 0:
        logger.log(
            f"Epoch {epoch:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

# End of training
logger.log("\nTraining finished!")
logger.log(f"Best validation loss: {best_val_loss:.4f}")

# Generate plots
logger.plot_metrics()
