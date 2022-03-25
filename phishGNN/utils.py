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

def tensor_to_tuple_list(tensor: torch.Tensor):
    assert len(tensor.shape) == 2, \
        "The tensor should be of shape (n, n)"
    
    edges = [(int(edge[0]), int(edge[1])) \
        for edge in zip(tensor[0], tensor[1])]
    return edges
    
def extract_domain_name(url):
    url = '{}://{}'.format(urlparse(url).scheme,
        urlparse(url).netloc)
    i = 1 if url.find('www.') != -1 else 0
    url = urlparse(url)
    return '.'.join(url.hostname.split('.')[i:])
