import os
import pathlib

from dataset import PhishingDataset

path = os.path.join(os.getcwd(), "dataset")
dataset = PhishingDataset(root=path)
