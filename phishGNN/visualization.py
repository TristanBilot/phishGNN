import collections
import glob
import os

import igraph
import matplotlib.pyplot as plt
import networkx as nx
import torch
from pyvis.network import Network
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from dataset_v1 import PhishingDataset
from utils.utils import extract_domain_name, tensor_to_tuple_list

ROOT_COLOR = '#0096FF'
DOMAIN_COLOR = '#73FCD6'
OUT_DOMAIN_COLOR = '#FFD479'
ERROR_COLOR = '#FF7E79'


def visualize(
        data: Data,
        width: int = 1000,
        height: int = 800,
        html_save_file: str = 'graph.html',
        generate_svg: bool = False,
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
    colors = []
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
        colors.append(node['color'])

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

    if generate_svg:
        g2 = igraph.Graph().from_networkx(G)
        g2 = g2.simplify()
        layout = g2.layout_auto()

        visual_style = {}
        visual_style['vertex_size'] = 10
        visual_style['vertex_color'] = colors
        visual_style['vertex_label_dist'] = 1
        visual_style['vertex_label_size'] = 8

        visual_style['edge_color'] = 'lightgrey'
        visual_style['edge_width'] = 1
        visual_style['edge_curved'] = 0.1

        visual_style['layout'] = layout
        visual_style['bbox'] = (500, 500)
        visual_style['margin'] = 40

        igraph.plot(g2, target=f'text{len(data.x)}.svg', **visual_style)

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


def generate_every_graphs() -> None:
    """Creates the visulaization graphs as html files
    for every example in the dataset (based on the files
    in data/processed).
    """
    path = os.path.join(os.getcwd(), 'data', 'train')
    data_files = sorted(glob.glob(os.path.join(path, 'processed', 'data_viz*')))

    if not os.path.exists(path) or len(data_files) == 0:
        print(f'No csv raw files found in {path}')

    dataset = PhishingDataset(
        root=path,
        do_data_preparation=False,
        visualization_mode=True,
    )
    dataset = dataset.shuffle()
    print(f'Start generating graphs...')
    for i, data in enumerate(tqdm(dataset, total=len(dataset))):
        visualize(data, html_save_file=f'visualization/graphs/graph{i}.html')

    print(f'Graphs successfully created.')


def plot_embeddings(model: torch.nn.Module, loader: DataLoader) -> None:
    color_list = ['red', 'green']
    embs = []
    colors = []
    for data in loader:
        pred = model(data.x, data.edge_index, data.batch)
        embs.append(model.embeddings)
        colors += [color_list[int(y)] for y in data.y]
    embs = torch.cat(embs, dim=0)

    xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
    plt.scatter(xs, ys, color=colors)
    plt.savefig('embeddings.png')


if __name__ == '__main__':
    generate_every_graphs()
