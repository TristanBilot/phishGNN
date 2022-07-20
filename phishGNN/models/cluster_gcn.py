import torch
import torch.nn.functional as F
from torch import device, Tensor
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, OptTensor


class ClusterGCN(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = None,
            hidden_channels: int = 32,
            out_channels: int = None,
            device: device = None,
            pooling_fn: callable = global_mean_pool,
            nb_layers: int = 6,
    ):
        super().__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(nb_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.embeddings = None

    def forward(self, x: Tensor, edge_index: Adj, batch: OptTensor) -> Tensor:
        x = x.to(dtype=torch.float32)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.pooling_fn(x, batch)
        self.embeddings = x

        return x

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for layer in self.batch_norms:
            layer.reset_parameters()
