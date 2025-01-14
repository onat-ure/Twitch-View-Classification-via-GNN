import os
import torch
from models.gnn_regressor import GNNRegressor
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import NeighborLoader
from utils.plotting import TrainingLogger
import torch.nn.functional as F


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


total_nodes = data.num_nodes
train_nodes = data.train_mask.sum().item()
val_nodes = data.val_mask.sum().item()

print("\nData Split Information:")
print(f"Total nodes: {total_nodes:,}")
print(f"Training nodes: {train_nodes:,} ({train_nodes/total_nodes*100:.1f}%)")
print(f"Validation nodes: {val_nodes:,} ({val_nodes/total_nodes*100:.1f}%)")


BATCH_SIZE = 1024
NUM_NEIGHBORS = [10, 10]  

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


model = GNNRegressor(num_features=data.x.size(1),
hidden_channels=128
).to(device)
initial_lr = 0.01
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=7,
    min_lr=1e-6,
    verbose=True
)


MAX_GRAD_NORM = 1.0

loss_fn = torch.nn.MSELoss()



eps = 1e-8  
data.y = torch.log(data.y + eps)




def train_epoch():
    model.train()
    total_loss = 0
    correct_predictions = 0
    nodes_processed = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        out = model(batch.x, batch.edge_index)
        batch_mask = batch.train_mask
        
        loss = loss_fn(out[batch_mask], batch.y[batch_mask])
        loss.backward()
        optimizer.step()
        
        
        pred_orig = torch.exp(out[batch_mask]) - eps
        true_orig = torch.exp(batch.y[batch_mask]) - eps
        
        relative_error = torch.abs(pred_orig - true_orig) / (true_orig + eps)
        
        threshold = 0.10  
        correct_predictions += (relative_error <= threshold).sum().item()
        
        batch_nodes = batch_mask.sum().item()
        total_loss += loss.item() 
        nodes_processed += batch_nodes
        
        del batch, out
    
    avg_loss = total_loss / len(train_loader) 
    accuracy = (correct_predictions / max(nodes_processed, 1)) * 100
    return avg_loss, accuracy

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    nodes_processed = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        
        batch_mask = batch.val_mask
        loss = loss_fn(out[batch_mask], batch.y[batch_mask])
        
        
        pred_orig = torch.exp(out[batch_mask]) - eps
        true_orig = torch.exp(batch.y[batch_mask]) - eps
        
        relative_error = torch.abs(pred_orig - true_orig) / (true_orig + eps)
        
        threshold = 0.10  
        correct_predictions += (relative_error <= threshold).sum().item()
        
        batch_nodes = batch_mask.sum().item()
        total_loss += loss.item()
        nodes_processed += batch_nodes
        
        del batch, out
    
    avg_loss = total_loss / len(val_loader) 
    accuracy = (correct_predictions / max(nodes_processed, 1)) * 100
    return avg_loss, accuracy


best_val_loss = float('inf')
patience = 15  
patience_counter = 0
n_epochs = 300  

logger = TrainingLogger()


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
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = evaluate()
    
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    
    logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
    
    
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
    
    
    pbar.set_description(
        f"Epoch {epoch:03d} | "
        f"LR: {current_lr:.2e} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    
    if epoch % 10 == 0:
        logger.log(
            f"Epoch {epoch:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )


logger.log("\nTraining finished!")
logger.log(f"Best validation loss: {best_val_loss:.4f}")


logger.plot_metrics()
