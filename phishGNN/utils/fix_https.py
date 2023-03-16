import pandas as pd


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def fix_https(path: str, dataset_og: str, dataset_filtered_file: str):
    f = open(dataset_filtered_file, 'r')
    dataset_filtered = f.readlines()

    dataset_og = pd.read_csv(path + dataset_og)
    dictionnary = {}
    filtered_dataset = []

    for url in dataset_og['url']:
        is_https = url.startswith('https://')
        url = remove_prefix(url, 'https://')
        url = remove_prefix(url, 'http://')
        url = remove_prefix(url, 'www.')
        dictionnary[url.strip()] = is_https

    for i, url in enumerate(dataset_filtered):
        if i == 0:
            continue
        url = remove_prefix(url, 'http://www.')
        url = remove_prefix(url, 'www.')
        is_https = dictionnary[url.strip()]
        filtered_dataset.append(f'http{"s" if is_https else ""}://{url}')

    with open(dataset_filtered_file, 'w') as file:
        file.write(''.join(list(filtered_dataset)))

    print(f'{len(filtered_dataset)} links converted successfully.')
    return dictionnary


if __name__ == '__main__':
    path = 'data/'
    dataset_og = 'phishing_dataset.csv'
    dataset_filtered = f'filtered_{dataset_og}'
    
    fix_https(path, dataset_og, dataset_filtered)
