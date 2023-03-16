from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests import Response


def remove_prefix(text:str, prefix:str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_request(url:str) -> Response:
    try:
        res = requests.get(url, timeout=(3, 30))
        print(res, url)
        return res
    except requests.RequestException:
        return None


def is_phishable(url: str) -> bool:
    res = get_request(url)
    if res is None:
        return False

    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    has_form = soup.find('form') is not None
    has_input = soup.find('input') is not None
    has_textarea = soup.find('textarea') is not None

    return has_form or has_input or has_textarea


def save_filtered_urls(i: int, filtered_urls: List[str], path: str) -> None:
    # https://docs.python.org/3/library/functions.html#open
    # 'w' open for writing, truncating the file first
    # 'a' open for writing, appending to the end of file if it exists
    with open(path, 'a') as file:
        file.write('\n'.join([str(i), *filtered_urls]))


def apply_prefix(url: str) -> str:
    is_https = url.startswith('https://')
    url = remove_prefix(url, 'https://')
    url = remove_prefix(url, 'http://')
    url = remove_prefix(url, 'www.')

    return f'http{"s" if is_https else ""}://www.{url}'


def filter(path, dataset):
    df = pd.read_csv(path + dataset)

    filtered_urls = []
    for i, url in enumerate(df['url']):
        url = apply_prefix(url)
        
        if is_phishable(url):
            filtered_urls.append(url)
        
        if i % 5 == 0:
            save_filtered_urls(i, filtered_urls, f'filtered_{dataset}')


assert apply_prefix('http://test.com') == 'http://www.test.com'
assert apply_prefix('https://test.com') == 'https://www.test.com'
assert apply_prefix('https://www.test.com') == 'https://www.test.com'
assert apply_prefix('www.test.com') == 'http://www.test.com'
assert apply_prefix('test.com') == 'http://www.test.com'


if __name__ == '__main__':
    path = 'data/'
    dataset = 'kaggle_PhishingAndLegitimateURLs.csv'

    # dataset = "phishtank_dataset_merged.csv"
    filter(path, dataset)
