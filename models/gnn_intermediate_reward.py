import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, ReLU, Dropout, Sequential

class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, dropout=0.2):
        super(GNNModel, self).__init__()

        # GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fully connected layers
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1)  # Output QED, a single value
        )

    def forward(self, x, edge_index, batch):
        # GCN layers with ReLU
        x = x.to(torch.float32)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling over nodes in each graph
        x = global_mean_pool(x, batch)

        # Fully connected layers for regression
        x = self.mlp(x)
        return x.view(-1)  # Flatten for regression output