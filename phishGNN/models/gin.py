import os.path as osp
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
import torch_geometric.transforms as T

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super().__init__()

        self.device = device
        self.to(device)

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv4 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv5 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x
        # if NLLLoss used must use softmax
        # return F.log_softmax(x, dim=-1)

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
