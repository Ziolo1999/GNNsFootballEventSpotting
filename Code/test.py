import os
os.chdir("Code")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mplsoccer.pitch import Pitch
from DataPreprocessing import Dataset
from FileFinder import find_files
import logging
from tqdm import tqdm
from DataManager import DataManager

logging.basicConfig(level=logging.INFO)
files = find_files("../football_games")
DM = DataManager(files)
player_violation = DM.player_violation()
player_violation
DM.read_games()
DM.datasets
for file in tqdm(files[0:1]):
    dataset = Dataset(1/5, file.home)
    dataset.open_dataset(file.datafile, file.metafile)

dataset.generate_encodings()
dataset._generate_all_features()
dataset.animate_game(interval=1)

