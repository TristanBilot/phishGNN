import os
import torch
from torch_geometric.loader import DataLoader

from dataset_v1 import PhishingDataset
from dataset_v2 import PhishingDataset2


def train_test_loader(use_process: bool):
    path = os.path.join(os.getcwd(), "data", "train")
    dataset = PhishingDataset2(root=path, use_process=use_process)
    dataset = dataset.shuffle()

    train_test = 0.7
    train_dataset = dataset[:int(len(dataset) * train_test)]
    test_dataset1 = dataset[int(len(dataset) * train_test):]

    test_path = os.path.join(os.getcwd(), "data", "test")
    test_dataset2 = PhishingDataset2(root=test_path, use_process=use_process)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader
