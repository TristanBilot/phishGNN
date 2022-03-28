import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from utils.utils import log_fail, log_success, normalize_www_prefix

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class PhishingDataset(Dataset):
    """Dataset containing both phishing and non-phishing
    website urls.
    """
    def __init__(
        self,
        root: str,
        use_process: bool=True,
        visulization_mode: bool=False,
        nan_value: float=-1.0,
        max_depth: int=1,
        test=False,
        transform=None,
        pre_transform=None,
    ):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.use_process = use_process
        self.visulization_mode = visulization_mode
        self.nan_value = nan_value
        self.max_depth = max_depth
        self.test = test
        super(PhishingDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """File name of the csv dataset.
        """
        return glob.glob(os.path.join(self.raw_dir, "*"))

    @property
    def processed_file_names(self):
        return [file + ".pt" for file in self.raw_file_names]

    @property
    def num_classes(self):
        return 2

    def file_name(self, idx: int):
        if self.visulization_mode:
            return f'data_viz_{idx}.pt'
        return f'data_{idx}.pt'

    def process(self):
        """Reads csv files in data/raw and preprocess so that output
        preprocessed files are written in data/processed folder.
        """
        if not self.use_process:
            return

        # loop over all files in `raw_file_names`
        for raw_path in self.raw_paths:
            df = self._read_csv(raw_path)
            df = self._normalize_features(df)

            root_urls = df[~df['is_phishing'].isin([self.nan_value])]['url']
            df = df.set_index('url')

            # loop over each root urls in the dataset
            for i, (_, url) in enumerate(tqdm(root_urls.items(), total=len(root_urls))):
                edge_index, x, _, y, viz_utils = self._build_tensors(url, df)

                self.data = Data(x=x, edge_index=edge_index, y=y)
                torch.save(self.data, os.path.join(self.processed_dir, f'data_{i}.pt'))

                # save another file with variables needed for visualization
                self.data.pos = viz_utils
                torch.save(self.data, os.path.join(self.processed_dir, f'data_viz_{i}.pt'))


    def len(self):
        return (len(os.listdir(self.processed_dir)) - 4) // 2


    def _read_csv(self, path: str) -> pd.DataFrame:
        """Opens the csv dataset as DataFrame and cast types.
        """
        date_parser = lambda c: pd.to_datetime(c, format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
        df = pd.read_csv(
            path,
            index_col=False,
            parse_dates=['domain_creation_date'],
            date_parser=date_parser,
        )

        # equilibrate dataset classes as 50/50% benign/phishing
        nb_phishing = len(df[df['is_phishing'] == 1])
        benign = df.index[(df['is_phishing'] == 0)][:nb_phishing]
        other = df.index[~(df['is_phishing'] == 0)]
        df = pd.concat([df.iloc[benign], df.iloc[other]])

        # cast object dtypes
        df['url'] = df['url'].astype('string')
        df['cert_country'] = df['cert_country'].astype('string')

        # remove useless features
        del df['status_code']
        del df['depth']
        del df['domain_creation_date']
        del df['cert_country'] # TODO: handle cert_country and sort by "dangerous" country
        return df


    def _normalize_features(self, df: pd.DataFrame):
        """Pre-processes every feature so that they are normalized
        adn scaled between 0 and 1 range.

        Args:
            df: the dataframe to normalize

        Returns:
            DataFrame: the normalized dataframe
        """
        def bool_to_int(col: pd.Series):
            def wrapper(x):
                if np.isnan(x) or x not in [True, False]:
                    return self.nan_value
                return 1 if x == True else 0
            return col.apply(wrapper)
        
        def min_max_scaling(col: pd.Series):
            min = col.min()
            max = col.max()
            return col.apply(lambda x: (x - min) / (max - min))

        def normalize_refs(refs: List[Dict]):
            refs = json.loads(refs)
            for ref in refs:
                ref['url'] = self._normalize_url(ref['url'])
            return refs

        # normalize int & float columns
        num_column_idxs = [i for i, dt in enumerate(df.dtypes) if dt in [int, float]]
        df.iloc[:, num_column_idxs] = \
            df.iloc[:, num_column_idxs].apply(min_max_scaling, axis=0)

        # replace NaN values for non-string columns
        df.iloc[:, num_column_idxs] = df.iloc[:, num_column_idxs].fillna(self.nan_value)

        # normalize bool columns
        bool_columns = ["is_phishing", "is_https", "is_ip_address", "is_error_page",
            "has_sub_domain", "has_at_symbol", "is_valid_html", "has_form_with_url",
            "has_iframe", "use_mouseover", "is_cert_valid", "has_dns_record",
            "has_whois", "path_starts_with_url"]
        df[bool_columns] = df[bool_columns].apply(bool_to_int, axis=0)

        # normalize urls
        df['refs'] = df['refs'].apply(normalize_refs)
        df['url'] = df['url'].apply(self._normalize_url)
        df = df.drop_duplicates(subset='url', keep='first')

        return df


    def _build_tensors(self, root_url: str, df: pd.DataFrame):
        """Builds the required tensors for one graph.
        Theses matrices will be then used for training the GNN.

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
                node = df.loc[url]
            except KeyError:
                node = self.error_page_node_feature

            refs = node.refs
            map_url_to_id(url)

            for i, edge in enumerate(refs):
                ref = edge['url']
                if (url, ref, i) in visited:
                    break
                if ref not in df.index:
                    error_pages.add(ref)
                map_url_to_id(ref)

                from_.append(url_to_id[url])
                to_.append(url_to_id[ref])
                edges_.append([1]) # should be edge features
                    
                is_anchor = ref == url
                if not is_anchor:
                    queue.append(ref)
                visited.add((url, ref, i))

            # remove url and refs
            features = node.drop("refs").drop("is_phishing")
            id_to_feat[url_to_id[url]] = features
        
        x = [id_to_feat[k] for k in sorted(id_to_feat)]
        visualization = {
            "url_to_id": url_to_id,
            "error_pages": error_pages,
        }

        return (
            torch.tensor([from_, to_]).type(torch.LongTensor),
            torch.tensor(x),
            torch.tensor(edges_),
            torch.tensor(df.loc[root_url].is_phishing),
            visualization,
        )
        

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.file_name(idx)))

        
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
