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
args = Args
collate_fn = collateGCN

train_dataset = CALFData(split="train", args=args)
validation_dataset = CALFData(split="validate", args=args)

train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

validate_loader = torch.utils.data.DataLoader(validation_dataset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

labels, targets, representations = next(iter(train_loader))

model = ContextAwareModel(num_classes=2, args=args)
criterion_segmentation = ContextAwareLoss(K=train_dataset.K_parameters)
criterion_spotting = SpottingLoss(lambda_coord=args.lambda_coord, lambda_noobj=args.lambda_noobj)
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, 
                            betas=(0.9, 0.999), eps=1e-07, 
                            weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

losses = trainer(train_loader, validate_loader,
                    model, optimizer, scheduler, 
                    [criterion_segmentation, criterion_spotting], 
                    [args.loss_weight_segmentation, args.loss_weight_detection],
                    model_name=args.model_name,
                    max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)


################################################################
#                         VISUALISE                            #
################################################################

from Visualiser import collateVisGCN, Visualiser

args = Args
collate_fn = collateVisGCN
model_path = "models/Testing_Model/model1.pth.tar"
model = torch.load(model_path)
visualiser = Visualiser(collate_fn, args, model, smoothing=True)
visualiser.plot_predictions(frame_threshold=5000, save_dir="plts/PredictionsPlot.png")
visualiser.visualize(frame_threshold=5000, save_dir="animations/PredictionsAnnotated.mp4", interval=60)
mAP = visualiser.calculate_MAP()




################################################################
#                         GConvLSTM                            #
################################################################

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class GConvGRUModel(torch.nn.Module):

    def __init__(self, num_classes=2, args=None):
        super(GConvGRUModel, self).__init__()
        input_channel, multiplier = args.input_channel, args.feature_multiplier*2
        self.r_conv_1 = GConvGRU(input_channel, 8*multiplier, 5)
        self.r_conv_2 = GConvGRU(8*multiplier, 16*multiplier, 4)
        self.r_conv_3 = GConvGRU(16*multiplier, 32*multiplier, 3)
        self.r_conv_4 = GConvGRU(32*multiplier, 76*multiplier, 2)
        self.linear = torch.nn.Linear(76*multiplier, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, edge_index, training=False):
        x = self.r_conv_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=training)

        x = self.r_conv_2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=training)

        x = self.r_conv_3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=training)

        x = self.r_conv_4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=training)

        x = self.linear(x)
        return self.softmax(x)

gru = GConvGRUModel(args=args)


collate_fn = collateGCN

train_dataset = CALFData(split="train", args=args)
validation_dataset = CALFData(split="validate", args=args)

train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

labels, targets, representations = next(iter(train_loader))
representations.x.shape
representations.edge_index
res = gru(representations.x, representations.edge_index)