import os

import torch
from dataset import PhishingDataset
from torch_geometric.loader import DataLoader

from visualization import visualize
from model import GCN


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data")
    dataset = PhishingDataset(root=path, use_process=False, visulization_mode=False)
    dataset = dataset.shuffle()

    train_test = 0.8
    train_dataset = dataset[:int(len(dataset) * train_test)]
    test_dataset = dataset[int(len(dataset) * train_test):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
    )
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.long())  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.


    for epoch in range(1, 171):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


    # for i in range(10):
    #     data = dataset[i]
    #     visualize(data, html_save_file=f"graphs/graph{i}.html")
    