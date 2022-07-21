import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import normalize_www_prefix

NAN_VALUE = -1


def read_csv(path: str) -> pd.DataFrame:
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
    del df['cert_country']  # TODO: handle cert_country and sort by "dangerous" country
    return df


def normalize_url(url: str):
    """Strips last / and normalize www prefix.
    """
    url = url.rstrip('/')
    url = normalize_www_prefix(url)
    return url


def normalize_features(df: pd.DataFrame):
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
                return NAN_VALUE
            return 1 if x == True else 0

        return col.apply(wrapper)

    def min_max_scaling(col: pd.Series):
        min = col.min()
        max = col.max()
        return col.apply(lambda x: (x - min) / (max - min))

    def normalize_refs(refs: List[Dict]):
        refs = json.loads(refs)
        for ref in refs:
            ref['url'] = normalize_url(ref['url'])
        return refs

    # normalize int & float columns
    num_column_idxs = [i for i, dt in enumerate(df.dtypes) if dt in [int, float]]
    df.iloc[:, num_column_idxs] = \
        df.iloc[:, num_column_idxs].apply(min_max_scaling, axis=0)

    # replace NaN values for non-string columns
    df.iloc[:, num_column_idxs] = df.iloc[:, num_column_idxs].fillna(NAN_VALUE)

    # normalize bool columns
    bool_columns = ['is_phishing', 'is_https', 'is_ip_address', 'is_error_page',
                    'has_sub_domain', 'has_at_symbol', 'is_valid_html', 'has_form_with_url',
                    'has_iframe', 'use_mouseover', 'is_cert_valid', 'has_dns_record',
                    'has_whois', 'path_starts_with_url']
    df[bool_columns] = df[bool_columns].apply(bool_to_int, axis=0)

    # normalize urls
    df['refs'] = df['refs'].apply(normalize_refs)
    df['url'] = df['url'].apply(normalize_url)
    df = df.drop_duplicates(subset='url', keep='first')

    return df


def load_every_urls_with_features(df: pd.DataFrame, path: str) -> Tuple[List, List]:
    """Returns a list of every urls in dataset (root urls + every refs)
    along with their features.
    """
    every_urls, X = [], []
    df_as_dict = df.to_dict('records')

    for row in tqdm(df_as_dict):
        every_urls.append(row['url'])
        features = [value for key, value in row.items() if key not in ['refs', 'is_phishing', 'url']]
        X.append(features)

    return every_urls, X


def load_train_set(csv_file: str) -> Tuple[pd.DataFrame, List[List], List[int]]:
    """Opens the csv file in `csv_file` and returns every
    features and label of each root url in the dataset.

    Returns:
        df: the opened and pre-processed dataset
        X: the list of features (list) of each root url
        y: the list of labels (int) of each root url
    """
    df = read_csv(csv_file)
    df = normalize_features(df)

    root_urls = df[~df['is_phishing'].isin([NAN_VALUE])]['url']
    df = df.set_index('url')

    X, y = [], []
    for i, (_, url) in enumerate(tqdm(root_urls.items(), total=len(root_urls))):
        data = df.loc[url]
        y.append(data.is_phishing)
        X.append(data.drop('refs').drop('is_phishing'))

    return df.reset_index(), X, y
