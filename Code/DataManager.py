# Code used by all notebooks
import os

from DataPreprocessing import Dataset
from FileFinder import MatchFile, find_files
from typing import Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

class DataManager():
    # class that parses list of files
    def __init__(self, files: Union[list[MatchFile], None]=None, alive: bool = False) -> None:
        #
        self.matches = []
        self.datasets = []
        self.alive = alive

        if files is None:
            files = find_files("../data/EC2020")

        self.files = files
        if len(self.files) == 0:
            logging.warn("No files provided")
    
        self.home, self.away = [], []
        for f in self.files:
            self.home.append(f.home)
            self.away.append(f.away)

        assert len(self.home) == len(self.files)
        assert len(self.home) == len(self.away)

    def read_games(self):
        """ Reads all games and provides list of matrices that describe each game
        """
        for f in tqdm(self.files, desc="Total file count"):
            logging.info(f"Reading file {f.datafile}")
            dataset = Dataset(1 / 5, f.home, self.alive)
            dataset.open_dataset(f.datafile, f.metafile)
            dataset.generate_encodings()
            player_violation = dataset._generate_all_features()
            if len(player_violation)>0:
                logging.error(f"Match {f.name} does not have 11 players in the {player_violation} fremes.")
            self.datasets.append(dataset.matrix)
            self.matches.append(f.name)
            del dataset
    
    def player_violation(self):
        """ Reads all games and verify if there are player violations
        """
        games_violated = []
        for f in tqdm(self.files, desc="Total file count"):
            logging.info(f"Reading {f.name} game")
            dataset = Dataset(1 / 5, f.home, self.alive)
            dataset.open_dataset(f.datafile, f.metafile)
            frame_nr = len(dataset._player_violation())
            if frame_nr>0:
                logging.warning(f"Match {f.name} does not have 11 players in {frame_nr} frames.")  
                games_violated.append(f.name)
        return games_violated
    
