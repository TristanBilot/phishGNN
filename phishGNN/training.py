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
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from visualization import visualize, plot_embeddings
from models import GCN_2, GCN_3, GIN, GAT, GraphSAGE, ClusterGCN, MemPool
from utils.utils import mean_std_error
from loader import train_test_loader, get_full_dataset
from cross_validation import cross_validation_with_val_set


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
    use_process: bool=False,
    should_plot_embeddings: bool=False,
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
    _, test_loader = train_test_loader(use_process)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(
        os.path.join(os.getcwd(), "weights", weights_file),
        map_location=torch.device('cpu'))
    model.eval()

    if should_plot_embeddings:
        plot_embeddings(model, test_loader)
        return

    correct = 0
    truths, preds = [], []
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        truths.extend(data.y.tolist())
        preds.extend(pred.tolist())
        
    print(confusion_matrix(truths, preds))
    accuracy = correct / len(test_loader.dataset)
    return accuracy



def train(
    use_process: bool=False,
):
    train_loader, test_loader = train_test_loader(use_process)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
        GAT,
        GIN,
        GCN_2,
        GCN_3,
        
        ClusterGCN,
        GraphSAGE,
        # MemPool,
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
            in_channels=train_loader.dataset.num_features,
            hidden_channels=neurons,
            out_channels=train_loader.dataset.num_classes,
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
        
    pprint(accuracies)



def train_cross_entropy(
    use_process: bool=False,
):
    dataset = get_full_dataset(use_process=use_process)

    models = [
        GAT,
        GIN,
        GCN_2,
        GCN_3,
        
        ClusterGCN,
        GraphSAGE,
        # MemPool,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

        loss_fn = torch.nn.CrossEntropyLoss()
        
        cross_validation_with_val_set(
            dataset,
            model,
            loss_fn,
            folds=5,
            epochs=11,
            batch_size=32,
            lr=.01,
            lr_decay_factor=.95,
            lr_decay_step_size=3,
            weight_decay=4e-5,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help='if set, a test will be run from data/test/raw')
    parser.add_argument('--process-dataset', action="store_true", help='if set, a training will be run from data/train/raw')
    parser.add_argument('--plot-embeddings', action="store_true",
        help='whether to save the embeddings in a png file during training or not')
    args, _ = parser.parse_known_args()

    if args.test:
        accuracy = test_model(
            "2_epochs/GCN_2_global_mean_pool_32.pkl",
            should_plot_embeddings=args.plot_embeddings)
        print(accuracy)
    else:
        train_cross_entropy(args.process_dataset)
