import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
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
        nb_layers=3,
    ):
        super(GCN, self).__init__()

        self.pooling_fn = pooling_fn
        self.device = device
        self.to(device)
        
        torch.manual_seed(12345)
        self.convs = ModuleList()
        for i in range(nb_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append( GCNConv(hidden_channels, hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)
        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.float()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = x.relu()

        x = self.pooling_fn(x, batch)
        self.embeddings = x

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        self.lin.reset_parameters()


class GCN_2(GCN):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(nb_layers=2, **kwargs)


class GCN_3(GCN):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(nb_layers=3, **kwargs)
