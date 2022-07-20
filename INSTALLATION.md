# create new miniconda environment 
```shell
conda create --name torchgraphs python=3.10 --channel conda-forge
conda activate torchgraphs
```

# install libs
```shell
conda install torchsparse pytorch_geometric matplotlib pyvis bs4 --channel conda-forge
pip install torch-scatter torch-cluster torch-spline-conv igraph
```

# unzip training/test data 
```shell
./install_dataset.sh
```

# run training
python phishGNN/training.py




### CRAWLER 
# setup MongoDb
