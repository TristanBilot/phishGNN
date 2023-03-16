import os
import sys
from typing import List, Tuple

import pandas as pd
import torch

sys.path.append(os.path.join(os.getcwd(), 'phishGNN'))

from dataset_v1 import PhishingDataset


def dataframe_mock(rows: List[Tuple[str, List, str]]):
    refs = [[{'url': ref, 'nb_edges': 1} for ref in row[1]] for row in rows]
    urls = [row[0] for row in rows]

    features = [
        # 'depth',
        'is_phishing',
        'redirects',
        'is_https',
        'is_ip_address',
        'is_error_page',
        'url_length',
        'domain_url_depth',
        'domain_url_length',
        'has_sub_domain',
        'has_at_symbol',
        'dashes_count',
        'path_starts_with_url',
        'is_valid_html',
        'anchors_count',
        'forms_count',
        'javascript_count',
        'self_anchors_count',
        'has_form_with_url',
        'has_iframe',
        'use_mouseover',
        'is_cert_valid',
        'has_dns_record',
        'has_whois',
        'cert_reliability',
        'domain_age',
        'domain_end_period',
    ]
    features = {feat: [-1 for _ in range(len(rows))] for feat in features}
    features['redirects'] = [row[2] for row in rows]
    data = {
        'url': urls,
        **features,
        'refs': refs,
    }
    df = pd.DataFrame(data=data)
    df = df.set_index('url')
    df.at[rows[0][0], 'is_phishing'] = 1

    return df


def test_dataset_easy1():
    df = dataframe_mock([
        ('root', ['a', 'b'], 2),
        ('a', [], 3),
        ('b', [], 4),
    ])
    idx_should_be = [[0., 0.], [1., 2.]]
    x_should_be = [2, 3, 4]

    dataset = PhishingDataset('root')
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_easy2():
    df = dataframe_mock([
        ('root', ['a', 'b'], 2),
        ('a', ['c', 'd'], 3),
        ('b', [], 4),
        ('c', [], 5),
        ('d', ['e'], 6),
        ('e', [], 7),
    ])
    idx_should_be = [[0., 0., 1., 1., 4.], [1., 2., 3., 4., 5.]]
    x_should_be = [2, 3, 4, 5, 6, 7]

    dataset = PhishingDataset('root')
    # FIXME: existing_urls is missing in the following call
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_loop1():
    df = dataframe_mock([
        ('root', ['a', 'b'], 2),
        ('a', ['root'], 3),
        ('b', [], 4),
    ])
    idx_should_be = [[0., 0., 1.], [1., 2., 0.]]
    x_should_be = [2, 3, 4]

    dataset = PhishingDataset('root')
    # FIXME: existing_urls is missing in the following call
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_loop2():
    df = dataframe_mock([
        ('root', ['root', 'b', 'root', 'root'], 2),
        ('a', ['root'], 3),
        ('b', ['root'], 4),
    ])
    idx_should_be = [[0., 0., 0., 0., 1.], [0., 1., 0., 0., 0.]]
    x_should_be = [2, 4]

    dataset = PhishingDataset('root')
    # FIXME: existing_urls is missing in the following call
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_missing1():
    df = dataframe_mock([
        ('root', ['a'], 2),
    ])
    idx_should_be = [[0.], [1.]]
    x_should_be = [2, -1]

    dataset = PhishingDataset('root')
    # FIXME: existing_urls is missing in the following call
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_missing2():
    df = dataframe_mock([
        ('root', ['a', 'b', 'c', 'd'], 2),
        ('a', ['root'], 3),
    ])
    idx_should_be = [[0., 0., 0., 0., 1.], [1., 2., 3., 4., 0.]]
    x_should_be = [2, 3, -1, -1, -1]

    dataset = PhishingDataset('root')
    # FIXME: existing_urls is missing in the following call
    edge_index, x, edge_attr, y, _ = dataset._build_tensors('root', df)

    assert torch.all(torch.eq(edge_index, torch.tensor(idx_should_be)))
    assert [int(i[0]) for i in x] == x_should_be
    assert y == 1.


def test_dataset_normalize_url():
    dataset = PhishingDataset('root')
    assert dataset._normalize_url('http://test.com') == 'http://www.test.com'
    assert dataset._normalize_url('https://test.com') == 'https://www.test.com'
    assert dataset._normalize_url('https://www.test.com') == 'https://www.test.com'
    assert dataset._normalize_url('www.test.com') == 'http://www.test.com'
    assert dataset._normalize_url('test.com') == 'http://www.test.com'
    assert dataset._normalize_url('http://www.test.com/') == 'http://www.test.com'
    assert dataset._normalize_url('http://www.test.com') == 'http://www.test.com'


test_dataset_easy1()
