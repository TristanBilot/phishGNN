import os
import itertools
import json
from collections import defaultdict
from pprint import pprint

import torch
import torch_geometric.nn as nn
from dataset import PhishingDataset
from torch_geometric.loader import DataLoader

from visualization import visualize, plot_embeddings
from models import GCN_2, GCN_3, GIN, GAT, GraphSAGE, ClusterGCN, MemPool
from utils.utils import mean_std_error


def fit(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += float(loss) * data.num_graphs
    
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data")
    dataset = PhishingDataset(root=path, use_process=False, visulization_mode=False)
    dataset = dataset.shuffle()

    train_test = 0.8
    train_dataset = dataset[:int(len(dataset) * train_test)]
    test_dataset = dataset[int(len(dataset) * train_test):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
        GAT,
        GIN,
        GCN_2,
        GCN_3,
        ClusterGCN,
        GraphSAGE,
        MemPool,
    ]

    poolings = [
        nn.global_mean_pool,
        nn.global_max_pool,
        nn.global_add_pool,
    ]

    hidden_neurons = [
        16,
        32,
        64,
    ]

    lr = 0.01
    weight_decay = 4e-5
    epochs = 20
    
    accuracies = defaultdict(lambda: [])
    for (model, pooling, neurons) in itertools.product(
        models,
        poolings,
        hidden_neurons,
    ):
        model = model(
            in_channels=dataset.num_features,
            hidden_channels=neurons,
            out_channels=dataset.num_classes,
            pooling_fn=pooling,
            device=device,
        )
        label = f"{model.__class__.__name__}_{pooling.__name__}_{neurons}"
        print(f"\n{label}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_accs, test_accs = [], []
        for epoch in range(epochs):
            if hasattr(model, 'fit'):
                loss = model.fit(train_loader, optimizer, loss_fn, device)
            else:
                loss = fit(model, train_loader, optimizer, loss_fn, device)
            if hasattr(model, 'test'):
                train_acc = model.test(train_loader)
                test_acc = model.test(test_loader)
            else:
                train_acc = test(model, train_loader)
                test_acc = test(model, test_loader)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print(f'Epoch: {(epoch+1):03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        accuracies[model.__class__.__name__].append({
            f'{pooling.__name__}, {neurons}': {
                'train': mean_std_error(train_accs),
                'test':  mean_std_error(test_accs),
            }
        })

        with open('training.logs', 'w') as logs:
            formatted = json.dumps(accuracies, indent=2)
            logs.write(formatted)
        
        os.makedirs("weights", exist_ok=True)
        torch.save(model, f"weights/{label}.pkl")
        
        # loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # plot_embeddings(model, loader)
    
    pprint(accuracies)
