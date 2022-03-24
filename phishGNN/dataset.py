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

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class PhishingDataset(Dataset):
    def __init__(self, root, nan_value: float=-1.0, max_depth: int=1, test=False, transform=None, pre_transform=None):
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
        idx = 0
        for raw_path in self.raw_paths:
            # loop over all files in `raw_file_names`

            df = self._read_csv(raw_path)
            df = self._normalize_features(df)

            root_urls = df[~df['is_phishing'].isnull()]['url']
            df = df.set_index('url')

            for _, url in root_urls.items():
                edge_index, x, _, y = self._build_tensors(url, df)

                data = Data(x=x, edge_index=edge_index, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1


    def len(self):
        return len(self.processed_file_names)


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
        # del df['depth']
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
            "has_whois"]
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
            Tuple[edge_index, x, edge_attr, y]
        """
        from_, to_, edges_ = [], [], []
        id_to_feat = {}
        idx, idxs = 0, {}
        queue = [root_url]

        while True:
            url = queue.pop()
            node = df.loc[url]
            refs = node.refs
            refs = json.loads(refs) \
                if refs is not np.nan else {}
            
            is_leaf = node.depth == self.max_depth
            if len(refs.keys()) > 0 and not is_leaf:
                queue.extend(refs.keys())
            idxs[url] = idx
            idx += 1

            for ref, edge_feats in refs.items():
                if ref not in idxs:
                    idxs[ref] = idx
                    idx += 1
                from_.append(idxs[url])
                to_.append(idxs[ref])
                edges_.append([1]) # should be edge features

            # remove url and refs
            features = node[1:-1]
            id_to_feat[idxs[url]] = features

            if len(queue) == 0:
                break
        
        x = [id_to_feat[k] for k in sorted(id_to_feat)] # (n, d)

        return (
            torch.tensor([from_, to_]),
            torch.tensor(x),
            torch.tensor(edges_),
            torch.tensor(df.loc[root_url]),
        )
        
        
        
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


