from DataManager import CALFData, DataManager, collateGCN
from FileFinder import MatchFile, find_files
from DataPreprocessing import DatasetPreprocessor
import numpy as np
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE, K_V2_ALIVE
from helpers.preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors, unproject_image_point, meter2radar
import torch 
import random
import copy
from tqdm import tqdm
from Model import ContextAwareModel
from helpers.loss import ContextAwareLoss, SpottingLoss
from train import trainer

from torch_geometric.data import Data
from torch_geometric.data import Batch
from dataclasses import dataclass
from torch_geometric.nn.conv import GCNConv
import logging
from torch.utils.data import Dataset

import matplotlib.animation as animation
from mplsoccer.pitch import Pitch
import logging
import seaborn as sns
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import os 

@dataclass
class Args:
    receptive_field = 20
    framerate = 5
    chunks_per_epoch = 1824
    class_split = "alive"
    num_detections = 200
    chunk_size = 60
    batch_size = 32
    input_channel = 10
    feature_multiplier=1
    backbone_player = "GCN"
    max_epochs=100
    load_weights=None
    model_name="Testing_Model"
    evaluation_frequency=20
    dim_capsule=16
    lambda_coord=5.0
    lambda_noobj=0.5
    loss_weight_segmentation=0.005
    loss_weight_detection=1.0
    patience=25
    LR=1e-03
    GPU=0 
    max_num_worker=1
    loglevel='INFO'

################################################################
#                           TRAIN                              #
################################################################
# args = Args
# collate_fn = collateGCN

# train_dataset = CALFData(split="train", args=args)
# validation_dataset = CALFData(split="validate", args=args)

# train_loader = torch.utils.data.DataLoader(train_dataset,
#             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# validate_loader = torch.utils.data.DataLoader(validation_dataset,
#             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# labels, targets, representations = next(iter(train_loader))

# model = ContextAwareModel(num_classes=2, args=args)
# criterion_segmentation = ContextAwareLoss(K=train_dataset.K_parameters)
# criterion_spotting = SpottingLoss(lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
#                             betas=(0.9, 0.999), eps=1e-07, 
#                             weight_decay=0, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

# losses = trainer(train_loader, validate_loader,
#                     model, optimizer, scheduler, 
#                     [criterion_segmentation, criterion_spotting], 
#                     [args.loss_weight_segmentation, args.loss_weight_detection],
#                     model_name=args.model_name,
#                     max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)


################################################################
#                         VISUALISE                            #
################################################################

class VisualiseDataset(Dataset):
    def __init__(self, args):
        
        listGames = find_files("../football_games")
        self.args = args

        DM = DataManager(files=listGames[0:1], framerate=args.framerate/25, alive=False)
        DM.read_games(ball_coords=True)
        DM.datasets[0].shape

        self.window = args.chunk_size*args.framerate
        self.windows_count = np.floor(DM.datasets[0].shape[0]/self.window)

        self.representation = []
        self.matrix = DM.datasets[0]
        self.ball_coords = DM.ball_coords

        for frame in range(DM.datasets[0].shape[0]):
            # Get nodes features
            Features = DM.datasets[0][frame].T
            # Get edges indicses
            rows, cols = np.nonzero(DM.edges[0][frame])
            Edges = np.stack((rows, cols))
            edge_index = torch.tensor(Edges, dtype=torch.long)
            x = torch.tensor(Features, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            self.representation.append(data)
    
    def __getitem__(self,index):
        indx = 300*index
        clip_representation = copy.deepcopy(self.representation[indx:indx+self.window])
        return clip_representation

    def __len__(self):
        return int(self.windows_count)

def collateGCN(list_of_examples):
    return Batch.from_data_list(list_of_examples[0])

class Visualiser():

    def __init__(self, collate_fn, args):   
        collate_fn = collate_fn
        data_visualise = VisualiseDataset(args=args)
        visualise_loader = torch.utils.data.DataLoader(data_visualise,
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        concatenated_seg =  torch.empty((0,2))
        concatenated_spot =  torch.empty((0,4))

        for representation in visualise_loader:
            segmentation, spotting = model(representation)
            reshaped_seg = torch.reshape(segmentation, (segmentation.shape[0]*segmentation.shape[1], segmentation.shape[2]))
            concatenated_seg = torch.cat((concatenated_seg, reshaped_seg), dim=0)
            reshaped_spot = torch.reshape(spotting, (spotting.shape[0]*spotting.shape[1], spotting.shape[2]))
            concatenated_spot = torch.cat((concatenated_spot, reshaped_spot), dim=0)

        self.segmentation = concatenated_seg.detach().numpy()
        self.spotting = concatenated_spot.detach().numpy()
        self.args = args
        self.matrix = data_visualise.matrix
        self.ball_coords = data_visualise.ball_coords[0]
    
    def visualize(self, frame_threshold=None, save_dir=None, interval=1):
        pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)

        # get scalars to represent players position on the map
        scalars = (pitch.dim.pitch_length, pitch.dim.pitch_width)
        coords = self.matrix.copy()
        coords[:,0,:] = coords[:,0,:]*scalars[0]
        coords[:,1,:] = coords[:,1,:]*scalars[1]
        ball_coords = self.ball_coords.copy()
        ball_coords[:,0] = ball_coords[:,0]*scalars[0]
        ball_coords[:,1] = ball_coords[:,1]*scalars[1]

        # create base animation
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        pitch.draw(ax=ax1)
        
        # create an empty collection for edges
        edge_collection = LineCollection([], colors='white', linewidths=0.5)
        # add the collection to the axis
        ax1.add_collection(edge_collection)

        # base scatter boxes
        scat_home = ax1.scatter([], [], c="r", s=50)
        scat_away = ax1.scatter([], [], c="b", s=50)
        scat_ball = ax1.scatter([], [], c="black", s=50)
        # base title
        timestamp = ax1.set_title(f"Timestamp: {0}")
        
        # Segmentation plot
        seg_alive = ax2.plot(np.arange(0, frame_threshold), self.segmentation[:frame_threshold,0], label='Alive')
        seg_dead = ax2.plot(np.arange(0, frame_threshold), self.segmentation[:frame_threshold,1], label='Dead')
        ax2.set_title(f"Segmentation")
        ax2.legend()

        # Spotting plot
        spot_alive = ax3.plot(np.arange(0, frame_threshold), self.spotting[:frame_threshold,2], label='Alive')
        spot_dead = ax3.plot(np.arange(0, frame_threshold), self.spotting[:frame_threshold,3], label='Dead')
        ax2.set_title(f"Segmentation")
        ax2.legend()

        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            seg_alive[0].set_data(np.arange(0, frame_threshold), self.segmentation[:frame_threshold,0])
            seg_dead[0].set_data(np.arange(0, frame_threshold), self.segmentation[:frame_threshold,1])
            spot_alive[0].set_data(np.arange(0, frame_threshold), self.spotting[:frame_threshold,2])
            spot_dead[0].set_data(np.arange(0, frame_threshold), self.spotting[:frame_threshold,3])
            return (scat_home,scat_away,scat_ball)
        
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])
            seg_alive[0].set_data(np.arange(0, frame + 1), self.segmentation[:frame+1, 0])
            seg_dead[0].set_data(np.arange(0, frame + 1), self.segmentation[:frame+1, 1])
            spot_alive[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, 2])
            spot_dead[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, 3])
            # convert seconds to minutes and seconds
            # minutes, seconds = divmod(self.game_details[frame], 60)
            # format the output as mm:ss
            # formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
            # timestamp.set_text(f"Timestamp: {formatted_time}")
            return (scat_home, scat_away, scat_ball)
        
        # get number of iterations
        if frame_threshold != None:
            iterartions = frame_threshold
        else:
            iterartions = self.matrix.shape[0]

        # set order of the plot components
        scat_home.set_zorder(3) 
        scat_away.set_zorder(3)
        scat_ball.set_zorder(3)

        # use animation 
        ani = animation.FuncAnimation(fig=fig, func=update, frames=iterartions, init_func=init, interval=interval)
        if save_dir != None:
            ani.save(save_dir, writer='ffmpeg') 
        else:
            plt.show()
        # delete data copies
        del coords
        del ball_coords

args = Args
collate_fn = collateGCN
model_path = "models/Testing_Model/model1.pth.tar"
model = torch.load(model_path)
visualiser= Visualiser(collate_fn, args)
visualiser.visualize(frame_threshold=5000, save_dir="TEST.mp4", interval=60)
visualiser.segmentation.shape
visualiser.spotting[2000:3000,2]

plt.plot(np.arange(0, 5000), visualiser.segmentation[:5000,0], label='Alive')
plt.show()