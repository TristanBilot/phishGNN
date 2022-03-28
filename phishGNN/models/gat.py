import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GAT, self).__init__()

        if hidden_channels is None:
            hidden_channels = 8

        self.device = device
        self.to(device)

        self.conv1 = GATConv(in_channels, hidden_channels, heads=hidden_channels, dropout=0.6)
        self.conv2 = GATConv(
            hidden_channels * hidden_channels, out_channels, heads=1, concat=False, dropout=0.6)
        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        self.embeddings = x

        return x
