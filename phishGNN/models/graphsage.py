import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GraphSAGE, self).__init__()

        self.device = device
        self.to(device)

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.embeddings = None

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

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