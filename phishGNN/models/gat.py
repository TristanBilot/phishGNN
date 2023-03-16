import torch
import torch.nn.functional as F
from torch import Tensor, device
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, OptTensor

class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels:int=None,
        hidden_channels:int=8,
        out_channels:int=None,
        pooling_fn:callable=global_mean_pool,
        device:device=None,
    ):
        super(GAT, self).__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)

        self.conv1 = GATConv(in_channels, hidden_channels, heads=hidden_channels, dropout=0.6)
        self.conv2 = GATConv(
            hidden_channels * hidden_channels, out_channels, heads=1, concat=False, dropout=0.6)
        self.embeddings = None

    def forward(self, x:Tensor, edge_index:Adj, batch:OptTensor) -> Tensor:
        x = x.to(dtype=torch.float32)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = self.pooling_fn(x, batch)
        self.embeddings = x

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
