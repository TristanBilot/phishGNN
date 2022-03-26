import os

from visualization import visualize
from dataset import PhishingDataset


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data")
    dataset = PhishingDataset(root=path, use_process=False)

    for i in range(10):
        data = dataset[i]
        visualize(data, html_save_file=f"graphs/graph{i}.html")
    