import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GNNRegressor(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, dropout=0.2):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels)
        
        
        self.dropout = torch.nn.Dropout(dropout)
        
        
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

    def forward(self, x, edge_index):
        
        x1 = self.conv1(x, edge_index)
        x1 = self.batch_norm1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.batch_norm3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)
        
        
        x = x1 + x2 + x3
        
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x.squeeze(-1)


