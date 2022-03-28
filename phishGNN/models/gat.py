import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GAT, self).__init__()

        self.device = device
        self.to(device)

        self.conv1 = GATConv(in_channels, hidden_channels, heads=hidden_channels, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
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
        # return F.log_softmax(x, dim=-1)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(dataset.num_features, dataset.num_classes).to(device)
# data = data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


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

    # @torch.no_grad()
    # def test(data):
    #     self.eval()
    #     out, accs = self(data.x, data.edge_index), []
    #     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #         acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    #         accs.append(acc)
    #     return accs


    # for epoch in range(1, 201):
    #     train(data)
    #     train_acc, val_acc, test_acc = test(data)
    #     print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #         f'Test: {test_acc:.4f}')