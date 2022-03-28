import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GCN, self).__init__()

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

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        self.embeddings = x

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

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
