import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    def __init__(
        self,
        in_channels=None,
        hidden_channels=None,
        out_channels=None,
        pooling_fn=global_mean_pool,
        device=None,
        nb_layers=None,
    ):
        super(MLP, self).__init__()

        self.pooling_fn = pooling_fn
        self.hidden_channels = hidden_channels
        self.device = device
        self.to(device)

        self.flatten = nn.Flatten()
        self.lin = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()


    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.flatten(x)
        x = nn.Linear(x.shape[1], 2 * self.hidden_channels)(x)
        x = self.relu(x)
        x = self.lin(x)
        x = self.relu(x)

        x = self.pooling_fn(x, batch)
        x = self.lin_out(x)
        return x
