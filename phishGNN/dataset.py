import json
from operator import index
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import numpy as np 
import os
import glob
from tqdm import tqdm

from utils import log_fail, log_success

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class PhishingDataset(Dataset):
    def __init__(
        self,
        root,
        process: bool=True,
        nan_value: float=-1.0, max_depth: int=1, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
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

    def download(self):
        pass

    def process(self):
        # return
        idx = 0
        for raw_path in self.raw_paths:
            # loop over all files in `raw_file_names`

            df = self._read_csv(raw_path)
            df = self._normalize_features(df)

            root_urls = df[~df['is_phishing'].isnull()]['url']
            df = df.set_index('url')

            for _, url in root_urls.items():
                edge_index, x, _, y, viz_utils = self._build_tensors(url, df)

                self.data = Data(x=x, edge_index=edge_index, y=y)
                self.data.pos = viz_utils

                if self.pre_filter is not None and not self.pre_filter(self.data):
                    continue

                if self.pre_transform is not None:
                    self.data = self.pre_transform(self.data)

                torch.save(self.data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
                if idx == 10:
                    break


    def len(self):
        return 100


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
        df['url'] = df['url'].astype('string')
        df['cert_country'] = df['cert_country'].astype('string')

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

        # def str_to_float(col: pd.Series):
        #     col = col.fillna(self.nan_value)
        #     s = set(col.unique())
        #     s = {e: i for i, e in enumerate(s)}
            
        #     col = col.apply(lambda x: s[x])
        #     return min_max_scaling(col)

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
        # bool_column_idxs = [i for i, dt in enumerate(df.dtypes) if dt == bool]
        df[bool_columns] = df[bool_columns].apply(bool_to_int, axis=0)

        # ignore url
        # no_url = df.iloc[: , 1:]
        # str_column_idxs = [i + 1 for i, dt in enumerate(no_url.iloc[: , 1:].dtypes) if dt == "string"]
        # no_url.iloc[:, str_column_idxs] = no_url.iloc[:, str_column_idxs].apply(str_to_float, axis=0)
        # df.iloc[: , 1:] = no_url

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
        idx, url_to_id = 0, {}
        queue = [root_url]
        visited = set()
        error_pages = set()

        while True:
            if len(queue) == 0:
                break
            url = queue.pop()
            try:
                node = df.loc[url]
                log_success(f'{url} found in features.')
            except KeyError:
                log_fail(f'{url} not found in features.')
                node = self.error_page_node_feature

            refs = node.refs
            refs = json.loads(refs) \
                if refs is not np.nan else {}
            
            if url not in url_to_id:
                url_to_id[url] = idx
                idx += 1

            for i, edge in enumerate(refs):
                ref = edge['url']
                # nb_hrefs = edge['nb_edges']
                if (url, ref, i) in visited:
                    break
                # try:
                #     node = df.loc[ref]
                #     log_success(f'{ref} found in features.')
                # except KeyError:
                #     log_fail(f'{ref} not found in features.')
                #     continue
                node = self.error_page_node_feature
                if ref not in df.index:
                    error_pages.add(ref)
                if ref not in url_to_id:
                    url_to_id[ref] = idx
                    idx += 1
                from_.append(url_to_id[url])
                to_.append(url_to_id[ref])
                edges_.append([1]) # should be edge features
                    
                is_anchor = ref == url
                if not is_anchor:
                    queue.append(ref)
                visited.add((url, ref, i))

            # remove url and refs
            features = node[:-1]
            id_to_feat[url_to_id[url]] = features
        
        x = [id_to_feat[k] for k in sorted(id_to_feat)] # (n, d)
        visualization = {
            "url_to_id": url_to_id,
            "error_pages": error_pages,
        }
        log_success(f'{root_url} processed.')

        return (
            torch.tensor([from_, to_]),
            torch.tensor(x),
            torch.tensor(edges_),
            torch.tensor(df.loc[root_url].is_phishing),
            visualization,
        )
        

    def get(self, idx):
        return torch.load(os.path.join(os.path.join(self.root, 'processed'), f'data_{idx}.pt'))

        
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
            'refs': "{}",
        }
        return pd.Series(data=data)


        # while len(queue) != 0:
        #     ref = queue.pop()
        #     node = df[ref]

        # for ref, edge_feats in refs.items():
        #     if ref not in idxs:
        #         idxs[ref] = idx
        #         idx += 1
        #     from_.append(idxs[url])
        #     to_.append(idxs[ref])
        #     edges_.append([1]) # should be edge features

        # # remove url and refs
        # features = row[2:-1]
        # id_to_feat[idxs[url]] = features





        
    # @property
    # def raw_file_names(self):
    #     """ If this file exists in raw_dir, the download is not triggered.
    #         (The download func. is not implemented here)  
    #     """
    #     return self.filename

    # @property
    # def processed_file_names(self):
    #     """ If these files are found in raw_dir, processing is skipped"""
    #     self.data = pd.read_csv(self.raw_paths[0]).reset_index()

    #     if self.test:
    #         return [f'data_test_{i}.pt' for i in list(self.data.index)]
    #     else:
    #         return [f'data_{i}.pt' for i in list(self.data.index)]
        

    # def download(self):
    #     pass

    # def process(self):
    #     self.data = pd.read_csv(self.raw_paths[0]).reset_index()
    #     featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    #     for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
    #         # Featurize molecule
    #         f = featurizer.featurize(mol["smiles"])
    #         data = f[0].to_pyg_graph()
    #         data.y = self._get_label(mol["HIV_active"])
    #         data.smiles = mol["smiles"]
    #         if self.test:
    #             torch.save(data, 
    #                 os.path.join(self.processed_dir, 
    #                              f'data_test_{index}.pt'))
    #         else:
    #             torch.save(data, 
    #                 os.path.join(self.processed_dir, 
    #                              f'data_{index}.pt'))
            

    # def _get_label(self, label):
    #     label = np.asarray([label])
    #     return torch.tensor(label, dtype=torch.int64)

    # def len(self):
    #     return self.data.shape[0]

    # def get(self, idx):
    #     """ - Equivalent to __getitem__ in pytorch
    #         - Is not needed for PyG's InMemoryDataset
    #     """
    #     if self.test:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_test_{idx}.pt'))
    #     else:
    #         data = torch.load(os.path.join(self.processed_dir, 
    #                              f'data_{idx}.pt'))        
    #     return data


