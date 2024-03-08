# Code used by all notebooks
import os
from DataPreprocessing import DatasetPreprocessor
from FileFinder import MatchFile, find_files
from typing import Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
import random
import os
import torch
import logging
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE, get_K_params
from helpers.preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors
from torch_geometric.data import Data
from torch_geometric.data import Batch
import copy
from dataclasses import dataclass
    
class CALFData(Dataset):
    def __init__(self, split="train", args=None): 
        '''
        Initialize the dataset. The main components are:
        - game_representation: list that includes the game representaion for each frame. 
          For each frame the torch.geometric.data.Data object is created. Uses nodes features and edges.
        - game_labels: list that includes the game labels which are shifts. They have shape of (nr_frames, nr_classes). 
          Shift values indicating the time differences between events for each action/event over various time frames. For example, 
          in the first column -5 values indicate that the first class occur in 5 frames. 
        - game_anchors: is the list consisting lists depending on the number of classes/events. Each sublist consist of lists that represent [game index, frame index, event class]. For example, 
          if the number of classes is 3, then the list will consist of 3 lists. In the first list there will be lists of [game index, frame index, 0].
        '''
        self.args = args
        # Gather football data from files
        logging.info("Preprocessing Data")
        self.listGames = find_files("../football_games")

        if split == "train":
            DM = DataManager(files=self.listGames[0:10], framerate=args.framerate/25, alive=False)
        elif split == "validate":
            DM = DataManager(files=self.listGames[10:12], framerate=args.framerate/25, alive=False)
        
        DM.read_games(generate_augmented_data=args.generate_augmented_data)

        # self.features = args.features
        self.chunk_size = args.chunk_size*args.framerate
        self.receptive_field = args.receptive_field*args.framerate
        self.chunks_per_epoch = args.chunks_per_epoch
        self.framerate = args.framerate


        if self.args.class_split == "alive":
            self.dict_event = EVENT_DICTIONARY_V2_ALIVE
            self.K_parameters = args.K_parameters*args.framerate 
            self.num_classes = args.annotation_nr
        
        self.num_detections = args.num_detections
        self.split=split
        
        # logging.info("Pre-compute clips")
        self.game_labels = list()
        self.game_representation = list()
        self.game_anchors = list()
        self.game_size = list()

        for i in np.arange(self.num_classes+1):
            self.game_anchors.append(list())

        game_counter = 0
        for game_indx in tqdm(range(len(DM.datasets)), desc="Get labels & features"):
            # calculate shifts 
            shifts = oneHotToShifts(np.array(DM.annotations[game_indx]), self.K_parameters.numpy())
            # calculate anchors
            anchors = getChunks_anchors(shifts, game_counter, self.K_parameters.numpy(), self.chunk_size, self.receptive_field)
            game_counter = game_counter+1
            
            # Generate pytorch-geometrics Data
            representation = []
            for frame in range(DM.datasets[game_indx].shape[0]):
                # Get nodes features
                Features = DM.datasets[game_indx][frame].T
                x = torch.tensor(Features, dtype=torch.float)

                # Get edges indicses
                rows, cols = np.nonzero(DM.edges[game_indx][frame])
                Edges = np.stack((rows, cols))
                edge_index = torch.tensor(Edges, dtype=torch.long)
                
                # Get edge weights
                edge_attr = torch.tensor(
                                        [
                                            [
                                            DM.edges[game_indx][frame][x, y],
                                            DM.velocity_diffs[game_indx][frame][x, y],
                                            DM.acceleration_diffs[game_indx][frame][x, y],
                                            DM.direction_diffs[game_indx][frame][x, y]
                                            ] for x, y in zip(rows, cols)
                                        ], 
                                        dtype=torch.float
                                    )
                                                    
                # Generate Data 
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                representation.append(data)

            # Get game representation
            self.game_representation.append(representation)
            self.game_labels.append(shifts)

            for anchor in anchors:
                self.game_anchors[anchor[2]].append(anchor)
            
            self.game_size.append(len(representation))            

    def __getitem__(self, index):
        '''
        Selects random class, then it is used to generate random event from that class. Then the anchor (main frame) is selected.
        Afterwards it generates random shift to get random boundries for video chunks. Set -1 to the frames outside the receptive fields,
        Then the clip targets are calculated where columns represents: the confidence (equals to 1, the frame id divided by total frames (normalised), the rest columns are one-hot encodings for each class.
        At the end the list of representation is selected from game representation.
        '''

        cntr = 0
        # Retrieve the game index and the anchor
        # class_selection = random.randint(0, self.num_classes-1)
        class_probs = [0.1/3] + [0.9/7]*7 + [0.1/3]*2
        class_selection = random.choices(np.arange(self.num_classes), weights=class_probs)[0]
        event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
        game_index = self.game_anchors[class_selection][event_selection][0]
        anchor = self.game_anchors[class_selection][event_selection][1]

        # Prevents size mismatch when events from the end of the game are selected
        while (anchor < self.chunk_size) or (anchor > self.game_size[game_index]-self.chunk_size):
            event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
            game_index = self.game_anchors[class_selection][event_selection][0]
            anchor = self.game_anchors[class_selection][event_selection][1]

        # Compute the shift for event chunks
        # TODO: Decide about shift depends on the results of the model
        shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
        # shift = np.random.randint(-self.chunk_size, 0)
        start = anchor + shift
        # Extract the clips
        clip_labels = copy.deepcopy(self.game_labels[game_index][start:start+self.chunk_size])

        # Put loss to zero outside receptive field
        clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1
        if np.all(clip_labels == -1):
            print("All -1 in clip_labels")
            
        # Get the spotting target
        clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]
        clip_representation = None
        clip_representation = copy.deepcopy(self.game_representation[game_index][start:start+self.chunk_size])
        cntr+=1
        return torch.from_numpy(clip_labels), torch.from_numpy(clip_targets), clip_representation
    
    def __len__(self):
        return self.chunks_per_epoch

def collateGCN(list_of_examples):
    # data_list = [x[0] for x in list_of_examples]
    # tensors = [x[1] for x in list_of_examples]
    return torch.stack([x[0] for x in list_of_examples], dim=0), \
            torch.stack([x[1] for x in list_of_examples], dim=0), \
            Batch.from_data_list([x for b in list_of_examples for x in b[2]])

class DataManager():
    # class that parses list of files
    def __init__(self, files: Union[list[MatchFile], None]=None, alive: bool = False, framerate:float=1/5) -> None:
        self.framerate = framerate

        self.matches = []
        self.datasets = []
        self.edges = []
        self.velocity_diffs = []
        self.acceleration_diffs = []
        self.direction_diffs = []

        self.alive = alive
        self.annotations = []
        
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

    def read_games(self, ball_coords:bool=False, focused_annotation=None, generate_augmented_data:bool=False):
        """ Reads all games and provides list features and edges matrices 
        """
        if ball_coords:
            self.ball_coords = []

        for f in tqdm(self.files, desc="Data preprocessing"):
            logging.info(f"Reading file {f.datafile}")
            # Opens dataset
            dataset = DatasetPreprocessor(self.framerate, f.name, self.alive)
            dataset._open_dataset(f.datafile, f.metafile, f.annotatedfile)
            
            # Generates node features
            player_violation = dataset._generate_node_features()
            if len(player_violation)>0:
                logging.warning(f"Match {f.name} does not have 11 players in the {len(player_violation)} frames.")
            
            # Generate edges and synchronise annotations
            dataset._generate_edges(threshold=None)
            dataset._synchronize_annotations(focused_annotation=focused_annotation)

            # Appends data
            self.datasets.append(dataset.matrix)
            self.edges.append(dataset.edges)
            self.matches.append(f.name)
            self.annotations.append(dataset.annotations)
            self.velocity_diffs.append(dataset.velocity_diff)
            self.acceleration_diffs.append(dataset.acceleration_diff)
            self.direction_diffs.append(dataset.direction_diff)

            # Generate augmented data
            if generate_augmented_data:
                mirror_pairs = [(True, False), (False, True), (True, True)]
                for x_mirror, y_mirror in mirror_pairs:

                    # Generate mirrored actions
                    player_violation = dataset._generate_node_features(x_mirror, y_mirror)
                    dataset._generate_edges(None)
                    # dataset._synchronize_annotations(focused_annotation=focused_annotation)
                    
                    # Add to dataset
                    self.datasets.append(dataset.matrix)
                    self.edges.append(dataset.edges)
                    self.matches.append(f.name)
                    self.annotations.append(dataset.annotations)
                    self.velocity_diffs.append(dataset.velocity_diff)
                    self.acceleration_diffs.append(dataset.acceleration_diff)
                    self.direction_diffs.append(dataset.direction_diff)
            
            if ball_coords:
                self.ball_coords.append(dataset.ball_coords)
            del dataset
    
    def player_violation(self):
        """ Reads all games and verify if there are player violations used only for EDA
        """
        games_violated = []
        for f in tqdm(self.files, desc="Player violation"):
            logging.info(f"Reading {f.name} game")
            dataset = DatasetPreprocessor(self.framerate, f.home, self.alive)
            dataset._open_dataset(f.datafile, f.metafile)
            frame_nr = len(dataset._player_violation())
            if frame_nr>0:
                logging.warning(f"Match {f.name} does not have 11 players in {frame_nr} frames.")  
                games_violated.append(f.name)
        return games_violated
    
    # def get_pytorch_dataset(self):
    #     # Create representation data

    #     data = Data(x=x, edge_index=edge_index.t().contiguous())
    
# @dataclass
# class Args:
#     receptive_field = 6
#     framerate = 5
#     chunks_per_epoch = 1824
#     class_split = "alive"
#     num_detections = 15
#     chunk_size = 30
#     batch_size = 32
#     input_channel = 13
#     feature_multiplier=1
#     backbone_player = "GCN"
#     max_epochs=180
#     load_weights=None
#     model_name="Testing_Model"
#     evaluation_frequency=20
#     dim_capsule=16
#     lambda_coord=5.0
#     lambda_noobj=0.5
#     loss_weight_segmentation=1.0
#     loss_weight_detection=0.0
#     patience=25
#     LR=1e-03
#     GPU=0 
#     max_num_worker=1
#     loglevel='INFO'
#     annotation_nr = 10
#     K_parameters = get_K_params(chunk_size)
#     focused_annotation = None
#     generate_augmented_data = True


'''
The order of features is as follows: 
x-position, y-position, distance to the ball, speed, x-direction, y-direction, movement direction (in radians), order of average position, team affiliation, red card flag, acceleration, avg velocity, avg acceleration.
There is also order of players in the matrix. First eleven are the Belgian players, the last 11 are the opponents players.

'''

# args = Args
# collate_fn = collateGCN
# train_dataset = CALFData(split="train", args=args)
# train_loader = torch.utils.data.DataLoader(train_dataset,
#             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# label, targets, representations = next(iter(train_loader))

# listGames = find_files("../football_games")
# DM = DataManager(files=listGames[1:2], framerate=args.framerate/25, alive=False)
# DM.read_games(generate_augmented_data=True)