import glob
import os

import pandas as pd
import torch
import torch_geometric
from torch import Tensor
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

import dataprep
from other_models import train_random_forest
from utils.compute_device import COMPUTE_DEVICE

print(f"Torch version: {torch.__version__}")
print(f"Compute device: {COMPUTE_DEVICE}")
print(f"Torch geometric version: {torch_geometric.__version__}")

# set default dtype, as MPS Pytorch does not support float64
torch.set_default_dtype(torch.float32)


class PhishingDataset2(Dataset):
    """This version implements the pre-classification step on root nodes with random forest
    """
    def __init__(
            self,
            root: str,
            do_data_preparation: bool = True,
            visulization_mode: bool = False,
            nan_value: float = -1.0,
            transform=None,
            pre_transform=None,
    ):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.do_data_preparation = do_data_preparation
        self.visulization_mode = visulization_mode
        self.nan_value = nan_value
        super(PhishingDataset2, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """File name of the csv dataset. """
        return glob.glob(os.path.join(self.raw_dir, "*"))

    @property
    def processed_file_names(self) -> list[str]:
        return [file + ".pt" for file in self.raw_file_names]

    @property
    def num_classes(self):
        return 2

    def file_name(self, idx: int) -> str:
        if self.visulization_mode:
            return f'data_viz_{idx}.pt'
        return f'data_{idx}.pt'

    def process(self) -> None:
        """Reads csv files in data/raw and preprocess so that output
        preprocessed files are written in data/processed folder.
        """
        if not self.do_data_preparation:
            return

        # loop over all files in `raw_file_names`
        for raw_path in self.raw_paths:
            df, X, y = dataprep.load_train_set(raw_path)
            df_eval, X_eval, y_eval = dataprep.load_train_set("data/test/raw/evaloutput.csv")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            X_test = [*X_test, *X_eval]
            y_test = [*y_test, *y_eval]

            forest, _ = train_random_forest(X_train, X_test, y_train, y_test)

            every_urls, every_features = dataprep.load_every_urls_with_features(df, raw_path)
            every_preds = forest.predict(every_features)

            root_urls = df[~df['is_phishing'].isin([self.nan_value])]['url']

            df.drop(df.iloc[:, 2:-1], inplace=True, axis=1)
            df["url"]: every_urls
            df["is_phishing_pred"] = every_preds

            df = df.set_index('url')
            df_to_dict = df.to_dict("index")

            # loop over each root urls in the dataset
            for i, (_, url) in enumerate(tqdm(root_urls.items(), total=len(root_urls))):
                edge_index, x, _, y, viz_utils = self._build_tensors(url, df_to_dict, df.index)

                self.data = Data(x=x, edge_index=edge_index, y=y)
                torch.save(self.data, os.path.join(self.processed_dir, f'data_{i}.pt'))

                # save another file with variables needed for visualization
                self.data.pos = viz_utils
                torch.save(self.data, os.path.join(self.processed_dir, f'data_viz_{i}.pt'))

    def len(self):
        return (len(os.listdir(self.processed_dir)) - 4) // 2

    def _build_tensors(self, root_url: str, df_to_dict, existing_urls) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
        """Builds the required tensors for one graph.
        These matrices will be then used for training the GNN.

        Args:
            df: the dataset of one graph as form of pandas daframe

        Returns:
            Tuple[edge_index, x, edge_attr, y, viz_utils]
        """
        from_, to_, edges_ = [], [], []
        id_to_feat = {}
        url_to_id = {}
        queue = [root_url]
        visited = set()
        error_pages = set()

        def map_url_to_id(url: str):
            url_to_id[url] = len(url_to_id) \
                if url not in url_to_id else url_to_id[url]

        def bool_to_float(value: bool):
            return 1. if value else 0.

        while True:
            if len(queue) == 0:
                break
            url = queue.pop()
            try:
                node = df_to_dict[url]
            except KeyError:
                node = self.error_page_node_feature

            refs = node['refs']
            map_url_to_id(url)

            for i, edge in enumerate(refs):
                ref = edge['url']
                is_same_domain = bool_to_float(edge['is_same_domain'])
                is_form = bool_to_float(edge['is_form'])
                is_anchor = bool_to_float(edge['is_anchor'])

                if (url, ref, i) in visited:
                    break
                if ref not in existing_urls:
                    error_pages.add(ref)
                map_url_to_id(ref)

                from_.append(url_to_id[url])
                to_.append(url_to_id[ref])
                edges_.append([1])  # should be edge features

                is_anchor = ref == url
                if not is_anchor:
                    queue.append(ref)
                visited.add((url, ref, i))

            # remove url and refs
            features = [node['is_phishing_pred']]
            id_to_feat[url_to_id[url]] = features

        x = [id_to_feat[k] for k in sorted(id_to_feat)]
        visualization = {
            "url_to_id": url_to_id,
            "error_pages": error_pages,
        }

        return (
            torch.tensor([from_, to_], dtype=torch.int64),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(edges_, dtype=torch.int64),
            torch.tensor(df_to_dict[root_url]['is_phishing'], dtype=torch.int64),
            visualization,
        )

    def get(self, idx):
        t = torch.load(os.path.join(self.processed_dir, self.file_name(idx)))
        t.x = t.x.to(dtype=torch.float32)
        t.y = t.y.to(dtype=torch.int64)
        t.edge = t.to(dtype=torch.int64)
        return t

    @property
    def error_page_node_feature(self):
        data = {
            'is_phishing': self.nan_value,
            'is_phishing_pred': self.nan_value,
            'refs': [],
        }
        return pd.Series(data=data)
