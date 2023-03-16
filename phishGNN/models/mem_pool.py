import torch
import torch.nn.functional as F
from torch import Tensor, device
from torch.nn import BatchNorm1d, LeakyReLU, Linear
from torch_geometric.nn import DeepGCNLayer, GATConv, MemPooling
from torch_geometric.typing import OptTensor, Adj, PairTensor


class MemPool(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = None,
            hidden_channels: int = 32,
            out_channels: int = None,
            pooling_fn: callable = None,
            device: device = None,
            dropout: float = 0.5
    ):
        super().__init__()

        self.device = device
        self.to(device)
        self.dropout = dropout

        self.lin = Linear(in_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(2):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout)
            norm = BatchNorm1d(hidden_channels)
            act = LeakyReLU()
            self.convs.append(
                DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout))

        self.mem1 = MemPooling(hidden_channels, 80, heads=5, num_clusters=10)
        self.mem2 = MemPooling(80, out_channels, heads=5, num_clusters=1)
        self.embeddings = None

    def forward(self, x: Tensor, edge_index: Adj, batch: OptTensor) -> PairTensor:
        x = x.to(dtype=torch.float32)
        x = self.lin(x)
        for conv in self.convs:
            x = conv(x, edge_index)

        x, S1 = self.mem1(x, batch)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout)
        x, S2 = self.mem2(x)
        self.embeddings = x.squeeze(1)

        return (
            F.log_softmax(x.squeeze(1), dim=-1),
            MemPooling.kl_loss(S1) + MemPooling.kl_loss(S2),
        )

    def fit(
            self,
            train_loader,
            optimizer,
            loss_fn,
            device,
    ):
        self.train()

        self.mem1.k.requires_grad = False
        self.mem2.k.requires_grad = False
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(self.device)
            out = self(data.x, data.edge_index, data.batch)[0]
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

        kl_loss = 0.
        self.mem1.k.requires_grad = True
        self.mem2.k.requires_grad = True
        optimizer.zero_grad()
        for data in train_loader:
            data = data.to(device=self.device)
            kl_loss += self(data.x, data.edge_index, data.batch)[1]

        kl_loss /= len(train_loader.dataset)
        kl_loss.backward()
        optimizer.step()

        return kl_loss

    @torch.no_grad()
    def test(self, loader, device):
        self.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            out = self(data.x, data.edge_index, data.batch)[0]
            pred = out.argmax(dim=-1)
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.mem1.reset_parameters()
        self.mem2.reset_parameters()
