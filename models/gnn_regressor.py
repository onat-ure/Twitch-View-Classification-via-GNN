import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GNNRegressor(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)  # Output layer for regression

    def forward(self, x, edge_index):
        # First GCN layer + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer + ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Final linear layer for regression
        x = self.lin(x)
        return x.squeeze(-1)  # Output shape: [num_nodes]


