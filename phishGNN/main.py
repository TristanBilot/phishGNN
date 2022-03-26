import os

from visualization import visualize
from dataset import PhishingDataset


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data")
    dataset = PhishingDataset(root=path)
    data = dataset[0]

    visualize(data)
    