import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels=None,
        hidden_channels=32,
        out_channels=None,
        pooling_fn=global_mean_pool,
        device=None,
    ):
        super(GCN, self).__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)
        
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = self.pooling_fn(x, batch)
        self.embeddings = x

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
