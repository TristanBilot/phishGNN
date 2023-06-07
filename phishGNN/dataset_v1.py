import glob
import os

import pandas as pd
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

import dataprep
from utils.compute_device import COMPUTE_DEVICE
from utils.utils import normalize_www_prefix

print(f'Torch version: {torch.__version__}')
print(f'Compute device: {COMPUTE_DEVICE}')
print(f'Torch geometric version: {torch_geometric.__version__}')

# set default dtype, as MPS Pytorch does not support float64
torch.set_default_dtype(torch.float32)


class PhishingDataset(Dataset):
    """Dataset containing both phishing and non-phishing website urls. """

    def __init__(
            self,
            root: str,
            do_data_preparation: bool = True,
            visualization_mode: bool = False,
            nan_value: float = -1.0,
            transform=None,
            pre_transform=None,
    ):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.do_data_preparation = do_data_preparation
        self.visualization_mode = visualization_mode
        self.nan_value = nan_value
        super(PhishingDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list[str]:
        """File name of the csv dataset. """
        return glob.glob(os.path.join(self.raw_dir, '*'))

    @property
    def processed_file_names(self) -> list[str]:
        return [file + '.pt' for file in self.raw_file_names]

    @property
    def num_classes(self):
        return 2

    def file_name(self, idx: int) -> str:
        if self.visualization_mode:
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
            df = dataprep.read_csv(raw_path)
            df = dataprep.normalize_features(df)

            root_urls = df[~df['is_phishing'].isin([self.nan_value])]['url']
            df = df.set_index('url')
            df_to_dict = df.to_dict('index')

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
            features = [v for k, v in sorted(node.items())
                        if k not in ['refs', 'is_phishing']]
            id_to_feat[url_to_id[url]] = features

        x = [id_to_feat[k] for k in sorted(id_to_feat)]
        visualization = {
            'url_to_id': url_to_id,
            'error_pages': error_pages,
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
        t.edge_index = t.edge_index.to(dtype=torch.int64)
        return t

    @property
    def error_page_node_feature(self):
        data = {
            # 'depth': self.nan_value,
            'is_phishing': self.nan_value,
            'redirects': self.nan_value,
            'is_https': self.nan_value,
            'is_ip_address': self.nan_value,
            'is_error_page': self.nan_value,
            'url_length': self.nan_value,
            'domain_url_depth': self.nan_value,
            'domain_url_length': self.nan_value,
            'has_sub_domain': self.nan_value,
            'has_at_symbol': self.nan_value,
            'dashes_count': self.nan_value,
            'path_starts_with_url': self.nan_value,
            'is_valid_html': self.nan_value,
            'anchors_count': self.nan_value,
            'forms_count': self.nan_value,
            'javascript_count': self.nan_value,
            'self_anchors_count': self.nan_value,
            'has_form_with_url': self.nan_value,
            'has_iframe': self.nan_value,
            'use_mouseover': self.nan_value,
            'is_cert_valid': self.nan_value,
            'has_dns_record': self.nan_value,
            'has_whois': self.nan_value,
            'cert_reliability': self.nan_value,
            'domain_age': self.nan_value,
            'domain_end_period': self.nan_value,
            'refs': [],
        }
        return pd.Series(data=data)

    def _normalize_url(self, url: str):
        url = url.rstrip('/')
        url = normalize_www_prefix(url)
        return url
