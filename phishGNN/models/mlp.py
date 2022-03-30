import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class MLP(nn.Module):
    def __init__(
        self,
        in_channels=None,
        hidden_channels=None,
        out_channels=None,
        pooling_fn=global_mean_pool,
        device=None,
        nb_layers=None,
    ):
        super(MLP, self).__init__()

        self.pooling_fn = pooling_fn
        self.hidden_channels = hidden_channels
        self.device = device
        self.to(device)

        self.flatten = nn.Flatten()
        self.lin = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()


    def forward(self, x, edge_index, batch):
        x = x.float()
        x = x[0]
        # only use first features (root url features)
        x = nn.Linear(x.shape[0], 2 * self.hidden_channels)(x)
        x = self.relu(x)
        x = self.lin(x)
        x = self.relu(x)

        x = self.lin_out(x)
        return x


    def fit(
        model,
        train_loader,
        optimizer,
        loss_fn,
        device,
        dont_use_graphs: bool=False,
    ):
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # only use first features (root url features)
            loss = loss_fn(out, data.y[0].long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss) * data.num_graphs
        
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(model, loader, device):
        model.eval()

        correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            # dim -1 instead of 1
            pred = out.argmax(dim=-1)
            correct += int((pred == data.y).sum())

        return correct / len(loader.dataset)
