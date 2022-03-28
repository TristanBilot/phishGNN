import argparse
import os
import glob

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dataset import PhishingDataset


def predict(url: str, weights_file: str) -> int:
    path = os.path.join(os.getcwd(), "data", "predict")
    data_files = sorted(glob.glob(os.path.join(path, "processed", "*")))
    if not os.path.exists(path) or len(data_files) == 0:
        raise FileNotFoundError(f'No files found in path {path}, please the crawler before.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PhishingDataset(root=path, use_process=True)
    data = dataset[0]
    data = data.to(device)

    model = torch.load(os.path.join(os.getcwd(), "weights/", weights_file)).to(device)
    model.eval()
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    return int(pred.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url', type=str, help='the url to predict (phishing/benign)')
    parser.add_argument('pkl_file', type=str, default="GCN_3_global_mean_pool_32.pkl",
        help='the path to the model weights (.pkl)')
    args = parser.parse_args()

    pred = predict(args.url, args.weights_file)
    
    if pred == 1:
        print("Phishing")
    else:
        print("Benign")
