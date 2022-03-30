import argparse
import os
import itertools
import json
import time
from collections import defaultdict
from pprint import pprint

import torch
import torch_geometric.nn as nn
from dataset_v1 import PhishingDataset
from dataset_v2 import PhishingDataset2
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
def test(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


def test_model(
    weights_file: str,
    use_process: bool=True,
) -> float:
    """Searches for crawled data as csv files in data/test/raw/
    and then load and process a dataset on these data in order
    to make predictions on every urls in that dataset.

    Args:
        weights_file: path from the weights folder to the pkl model
        use_process: whether to run data process or not

    Returns:
        float: the accuracy of the model on these data
    """
    path = os.path.join(os.getcwd(), "data", "test")
    dataset = PhishingDataset(root=path, use_process=use_process)

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        os.path.join(os.getcwd(), "weights", weights_file),
        map_location=torch.device('cpu'))
    model.eval()

    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    
    accuracy = correct / len(loader.dataset)
    return accuracy



def train(
    should_plot_embeddings: bool=False,
    use_process: bool=False,
):
    path = os.path.join(os.getcwd(), "data", "train")
    dataset = PhishingDataset2(root=path, use_process=use_process)
    dataset = dataset.shuffle()

    train_test = 0.9
    train_dataset = dataset[:int(len(dataset) * train_test)]
    test_dataset1 = dataset[int(len(dataset) * train_test):]

    test_path = os.path.join(os.getcwd(), "data", "test")
    test_dataset2 = PhishingDataset2(root=test_path, use_process=use_process)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
        MLP,
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
    epochs = 10
    
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
        model = model.to(device)
        label = f"{model.__class__.__name__}_{pooling.__name__}_{neurons}"
        print(f"\n{label}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        start = time.time()
        train_accs, test_accs = [], []
        for epoch in range(epochs):
            if hasattr(model, 'fit'):
                loss = model.fit(train_loader, optimizer, loss_fn, device)
            else:
                loss = fit(model, train_loader, optimizer, loss_fn, device)
            if hasattr(model, 'test'):
                train_acc = model.test(train_loader, device)
                test_acc = model.test(test_loader, device)
            else:
                train_acc = test(model, train_loader, device)
                test_acc = test(model, test_loader, device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print(f'Epoch: {(epoch+1):03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        end = time.time()
        accuracies[model.__class__.__name__].append({
            f'{pooling.__name__}, {neurons}': {
                'train': mean_std_error(train_accs),
                'test':  mean_std_error(test_accs),
                'time': end - start,
            }
        })

        out_path = os.path.join("weights", f"{epochs}_epochs")
        os.makedirs(out_path, exist_ok=True)

        with open(os.path.join(out_path, f"accuracies_{epochs}_epochs.json"), 'w') as logs:
            formatted = json.dumps(accuracies, indent=2)
            logs.write(formatted)
        
        torch.save(model, f"{out_path}/{label}.pkl")
        
        if should_plot_embeddings:
            loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            plot_embeddings(model, loader)
    
    pprint(accuracies)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--training', action="store_true", help='if set, a training will be run from data/train/raw')
    # parser.add_argument('--test', action="store_true", help='if set, a test will be run from data/test/raw')
    # parser.add_argument('--plot-embeddings', action="store_true",
    #     help='whether to save the embeddings in a png file during training or not')
    # args, _ = parser.parse_known_args()

    # if args.test is not None:
    #     accuracy = test_model("20_epochs_default/ClusterGCN_global_max_pool_32.pkl")
    #     print(accuracy)
    # else:
    #     train(args.plot_embeddings)
    train(use_process=True)
