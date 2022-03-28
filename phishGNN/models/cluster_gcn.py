import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.nn import global_mean_pool


class ClusterGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, num_layers=6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.float()
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        self.embeddings = x

        return x

    def fit(
        self,
        train_loader,
        optimizer,
        loss_fn,
        device,
    ):
        self.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            out = self(data.x, data.edge_index, data.batch)
            loss = loss_fn(out, data.y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss) * data.num_graphs
        
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(self, loader):
        self.eval()

        correct = 0
        for data in loader:
            out = self(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())

        return correct / len(loader.dataset)
