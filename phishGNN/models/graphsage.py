import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor, device
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import OptTensor, Adj


class GraphSAGE(nn.Module):
    def __init__(
            self,
            in_channels: int = None,
            hidden_channels: int = 16,
            out_channels: int = None,
            pooling_fn: callable = global_mean_pool,
            device: device = None,
    ):
        super(GraphSAGE, self).__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.embeddings = None

    def forward(self, x: Tensor, edge_index: Adj, batch: OptTensor) -> Tensor:
        x = x.to(dtype=torch.float32)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = self.pooling_fn(x, batch)
        self.embeddings = x

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
