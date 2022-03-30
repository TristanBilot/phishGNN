# PhishGNN

Phishing website detection using Graph Neural Networks (GNNs).

<img width="40%" alt="phishing_graph" src="https://user-images.githubusercontent.com/40337775/160821966-c9c53dfe-8c54-4390-ac47-24975047d87a.png">

## Installation

### Clone the repo

```
git clone https://github.com/TristanBilot/phishGNN.git
cd phishGNN
```

### Install dependencies

```python
./install_dataset.sh
pip install -r requirements.txt
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html # for cpu
```

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

To visualize these data, first follow the instructions in the installation part and open the file `visualization/visualization.html`.

<center>
    <img width="75%" alt="Screenshot 2022-03-30 at 12 39 01" src="https://user-images.githubusercontent.com/40337775/160822019-712227d8-e000-4781-b55d-8b089409d53d.png">
</center>


## License
  <a href="https://opensource.org/licenses/MIT">MIT</a>
  