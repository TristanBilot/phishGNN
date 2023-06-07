from typing import List
from urllib.parse import urlparse

import numpy as np
import torch


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_success(msg: str):
    print(f'{bcolors.OKGREEN}SUCCESS:\t{bcolors.ENDC}{msg}')


def log_fail(msg: str):
    print(f'{bcolors.FAIL}FAILURE:\t{bcolors.ENDC}{msg}')


def tensor_to_tuple_list(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """Converts a tensor of shape [[x], [y]] in an
    array of tuples of shape [(x, y)].
    """
    assert len(tensor.shape) == 2, \
        "The tensor should be of shape (n, n)"

    edges = [(int(edge[0]), int(edge[1])) for edge in zip(tensor[0], tensor[1])]
    return edges


def extract_domain_name(url: str):
    """Returns only the domain part of an url.
    """
    url = '{}://{}'.format(urlparse(url).scheme,
                           urlparse(url).netloc)
    i = 1 if url.find('www.') != -1 else 0
    url = urlparse(url)
    return '.'.join(url.hostname.split('.')[i:])


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def normalize_www_prefix(url: str):
    is_https = url.startswith('https://')
    url = remove_prefix(url, 'https://')
    url = remove_prefix(url, 'http://')
    url = remove_prefix(url, 'www.')

    return f'http{"s" if is_https else ""}://www.{url}'


def mean_std_error(vals: List):
    mean = np.mean(vals)
    std = np.std(vals)
    std_error = std / np.sqrt(len(vals))
    return mean, std_error
