import argparse
import glob
import os

import torch

from dataset import PhishingDataset
from utils.compute_device import COMPUTE_DEVICE


def predict(weights_file: str) -> int:
    path = os.path.join(os.getcwd(), 'data', 'predict')
    data_files = sorted(glob.glob(os.path.join(path, 'raw' '*')))
    if not os.path.exists(path) or len(data_files) == 0:
        raise FileNotFoundError(f'No files found in path {path}, please generate a csv dataset from the URLs you want to predict'
            f' and move this file into {path}')

    dataset = PhishingDataset(root=path, predict=True, do_data_preparation=True)
    data = dataset.data
    data = data.to(COMPUTE_DEVICE)

    model = torch.load(os.path.join(os.getcwd(), 'weights/', weights_file)).to(COMPUTE_DEVICE)
    model.eval()
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    return int(pred.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default='GCN_3_global_mean_pool_32.pkl',
                        help='the path to the model weights (.pkl)')
    args = parser.parse_args()

    pred = predict(args.pkl_file)

    if pred == 1:
        print('Phishing')
    else:
        print('Benign')
