from typing import Dict, Set

from urllib.parse import urlparse
from pyvis.network import Network
import networkx as nx
import torch

from utils import tensor_to_tuple_list


ROOT_COLOR          = '#0096FF'
DOMAIN_COLOR        = '#73FCD6'
OUT_DOMAIN_COLOR    = '#FFD479'
ERROR_COLOR         = '#FF7E79'


def handle_error(error, error_obj, r, url, visited, error_codes):
    error = str(error_obj) if error else r.status_code
    visited.add(url)
    error_codes[url] = error
    print(f'{error} ERROR while visiting {url}')


def get_node_data(nodes, error_codes, resource_pages, args):
    data = []
    for node in nodes:
        if node in error_codes:
            data.append(f'Error: {error_codes[node]}')
        elif node in resource_pages:
            data.append('resource')
        elif node.startswith(args.site_url):
            data.append('internal')
        else:
            data.append('external')
    return data


def extract_domain_name(url):
    url = '{}://{}'.format(urlparse(url).scheme,
        urlparse(url).netloc)
    i = 1 if url.find('www.') != -1 else 0
    url = urlparse(url)
    return '.'.join(url.hostname.split('.')[i:])


def visualize(
    edge_index: torch.Tensor,
    url_to_id: Dict[str, int],
    error_pages: Set[str],
    width: int=1000,
    height: int=800,
    html_save_file: str="graph.html",
    with_features=False,
):
    id_to_url = {v: k for k, v in url_to_id.items()}
    G = nx.MultiDiGraph()
    edges = tensor_to_tuple_list(edge_index)
    # edges = [(x, y) for x, y in edges if x == 0]

    for x,y in edges:
        G.add_edge(x, y)

    net = Network(width=width, height=height, directed=True)
    net.from_nx(G)

    domain = extract_domain_name(id_to_url[0])
    for node in net.nodes:
        node_url = id_to_url[node['id']]
        node['size'] = 15
        node['label'] = ''
        if domain in node_url:
            node['color'] = DOMAIN_COLOR
        else:
            node['color'] = OUT_DOMAIN_COLOR
        if node['id'] == 0:
            node['color'] = ROOT_COLOR
        if node_url in error_pages:
            node['color'] = ERROR_COLOR

        node['title'] = f'<a href="{id_to_url[node["id"]]}">{id_to_url[node["id"]]}</a>'

    occurences = {}
    for x,y in edges:
        if (x, y) not in occurences:
            occurences[(x, y)] = 0
        occurences[(x, y)] += 1

    for e in net.edges:
        t = (e['from'], e['to'])
        nb_occurences = 0
        if t not in occurences:
            nb_occurences = occurences[(e['to'], e['from'])]
        else:
            nb_occurences = occurences[t]
        if nb_occurences > 1:
            e['label'] = nb_occurences

    net.save_graph(html_save_file)
