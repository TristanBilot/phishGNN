import collections
import glob
import os
from typing import Dict, Set

import matplotlib.pyplot as plt
import networkx as nx
import torch
from pyvis.network import Network
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from tqdm import tqdm

from dataset import PhishingDataset
from utils.utils import extract_domain_name, tensor_to_tuple_list

ROOT_COLOR          = '#0096FF'
DOMAIN_COLOR        = '#73FCD6'
OUT_DOMAIN_COLOR    = '#FFD479'
ERROR_COLOR         = '#FF7E79'


def visualize(
    data: Data,
    width: int=1000,
    height: int=800,
    html_save_file: str="graph.html",
):
    """Create an html file with the corresponding graph
    plotted using the pyvis library.
    """

    folder = os.path.dirname(html_save_file)
    if folder != '':
        os.makedirs(folder, exist_ok=True)

    edge_index = data.edge_index
    viz_utils = data.pos
    id_to_url = {v: k for k, v in viz_utils['url_to_id'].items()}
    edges = tensor_to_tuple_list(edge_index)
    # edges = [(x, y) for x, y in edges if x == 0]

    G = nx.MultiDiGraph()
    G.add_edges_from(edges)

    net = Network(width=width, height=height, directed=True)
    net.from_nx(G)

    root_url = id_to_url[0]
    domain = extract_domain_name(root_url)
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
        if node_url in viz_utils['error_pages']:
            node['color'] = ERROR_COLOR

        node['title'] = f'<a href="{id_to_url[node["id"]]}">{id_to_url[node["id"]]}</a>'

    count_edges = dict(collections.Counter(edges))
    for e in net.edges:
        t = (e['from'], e['to'])
        nb_occurences = 0
        if t not in count_edges:
            nb_occurences = count_edges[(e['to'], e['from'])]
        else:
            nb_occurences = count_edges[t]
        if nb_occurences > 1:
            e['label'] = nb_occurences

    net.save_graph(html_save_file)

    with open(html_save_file, 'a') as html_file:
        graph_data_html = f"""
            <div id="graph_data" 
                is_phishing="{data.y == 1.}"
                url="{root_url}"
                nb_edges="{data.num_edges}"
                nb_nodes="{data.num_nodes}"
            >
            </div>
        """
        html_file.write(graph_data_html)


def generate_every_graphs():
    """Creates the visulaization graphs as html files
    for every example in the dataset (based on the files
    in data/processed).
    """
    path = os.path.join(os.getcwd(), "data")
    data_files = sorted(glob.glob(os.path.join(path, "processed", "data_viz*")))
    use_process = False

    if not os.path.exists(path) or len(data_files) == 0:
        print(f"Warning: no data files found in {path}, processing the dataset...")
        raw_files = sorted(glob.glob(os.path.join(path, "raw", "*")))
        if len(raw_files) == 0:
            raise FileNotFoundError(f"No csv raw files found in {os.path.join(path, 'raw')}")
        print(f"{len(raw_files)} file(s) found in {os.path.join(path, 'raw')}")
        use_process = True

    dataset = PhishingDataset(
        root=path,
        use_process=use_process,
        visulization_mode=True,
    )
    dataset = dataset.shuffle()
    print(f"Start generating graphs...")
    for i, data in enumerate(tqdm(dataset, total=len(dataset))):
        visualize(data, html_save_file=f"graphs/graph{i}.html")

    print(f"Graphs successfully created.")


def plot_embeddings(
    model,
    loader,
):
    color_list = ["red", "green"]
    embs = []
    colors = []
    for data in loader:
        pred = model(data.x, data.edge_index, data.batch)
        embs.append(model.embeddings)
        colors += [color_list[int(y)] for y in data.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
    plt.scatter(xs, ys, color=colors)
    plt.savefig("embeddings.png")


if __name__ == "__main__":
    generate_every_graphs()
