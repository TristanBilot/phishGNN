# PhishGNN

Phishing website detection using Graph Neural Networks (GNNs).

<p float="left">
    <img width="35%" alt="phishing_graph" src="https://user-images.githubusercontent.com/40337775/165151501-0c0f37b0-c055-4085-b640-3a86e4c9a7d8.svg">
    <img width="35%" alt="phishing_graph" src="https://user-images.githubusercontent.com/40337775/165151748-4dca6de8-104f-4f1b-b03e-9054a1e399f4.svg">
</p>
    
## Installation

### Clone the repo

```
git clone https://github.com/TristanBilot/phishGNN.git
cd phishGNN
```

### Install dependencies

```python
python3 -m venv venv
. venv/bin/activate
pip install wheel
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html # for cpu
```

### unzip the dataset
```shell
./install_dataset.sh
```

## Dataset & crawler

The dataset can be downloaded in PyG format and new features can be extracted from URLs using the crawler.
A full guide for both tasks can be found <a href="https://tristanbilot.me/phishgnn">here</a>.

## Training

During training, the files located in data/training/processed will be used by default. The raw dataset is composed of urls mapped to around 30 features, including a list of references (href, form, iframe) to other pages, which also have their own features and their list of references.

```
python phishGNN/training.py
```

## Visualize node embeddings

During training, it is possible to generate the embeddings just after passing through the Graph Convolutional layers. Just run the training with the following option:

```
python phishGNN/training.py --plot-embeddings
```

<center>
    <img src="https://user-images.githubusercontent.com/40337775/160821779-8a6651c3-d4c0-4eca-bcd5-90910f35e766.png" width="55%"/>
</center>


<!-- <center>
<img src="embeddings.png" width="60%">
</center> -->

## Visualize the graphs
A tool has been developed in order to visualize graphically the internal structure of web pages from the dataset along with their characteristics such as the number of nodes/edges and whether the page is phishing or benign.

To visualize these data, first follow the instructions in the installation part, run the `visualization` script and open the file `visualization/visualization.html`.

```bash
python visualization.py
```

<center>
    <img width="75%" alt="Screenshot 2022-03-30 at 12 39 01" src="https://user-images.githubusercontent.com/40337775/160822019-712227d8-e000-4781-b55d-8b089409d53d.png">
</center>


## License
  <a href="https://opensource.org/licenses/MIT">MIT</a>
  
