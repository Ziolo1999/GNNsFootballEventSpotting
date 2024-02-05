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

    backbone_feature = None
    backbone_player = "GCN"
    tiny=None

    max_epochs=100
    load_weights=None
    model_name="Testing_Model"
    mode=0
    test_only=False
    challenge=True
    teacher=False
    K_params=None
    num_features=512
    evaluation_frequency=20
    dim_capsule=16
    lambda_coord=5.0
    lambda_noobj=0.5
    loss_weight_segmentation=0.005
    loss_weight_detection=1.0
    feature_multiplier=1
    calibration=False
    calibration_field=False
    calibration_cone=False
    calibration_confidence=False
    dim_representation_w=64
    dim_representation_h=32
    dim_representation_c=3
    dim_representation_player=2
    dist_graph_player=25
    with_dropout=0.0
    LR=1e-03
    patience=25
    GPU=0 
    max_num_worker=1
    loglevel='INFO'

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
                model, optimizer, scheduler, [criterion_segmentation, criterion_spotting], [args.loss_weight_segmentation, args.loss_weight_detection],
                model_name=args.model_name,
                max_epochs=args.max_epochs, evaluation_frequency=args.evaluation_frequency)

files = find_files("../football_games")
dataset = DatasetPreprocessor(1/5, files[15].name)
dataset.open_dataset(files[15].datafile, files[15].metafile)
dataset.dataset.frames[1388].timestamp
60*45*5