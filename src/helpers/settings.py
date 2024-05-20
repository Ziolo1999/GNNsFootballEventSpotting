from dataclasses import dataclass
import numpy as np
from helpers.classes import get_K_params

@dataclass
class Args:
    # DATA
    chunk_size = 60
    batch_size = 32
    input_channel = 13
    annotation_nr = 10
    receptive_field = 12
    fps = 5
    K_parameters = get_K_params(chunk_size)
    focused_annotation = None
    generate_augmented_data = True
    class_split = "alive"
    generate_artificial_targets = False
    
    # TRAINING
    chunks_per_epoch = 1824
    lambda_coord=5.0
    lambda_noobj=0.5
    patience=25
    LR=1e-03
    max_epochs=180
    GPU=0 
    max_num_worker=1
    loglevel='INFO'
    
    # SEGMENTATION MODULE
    feature_multiplier=1
    backbone_player = "GCN"
    load_weights=None
    model_name="Testing_Model"
    dim_capsule=16
    vocab_size=64
    pooling=None

    # SPOTTING MODULE
    sgementation_path = None
    freeze_model = None
    spotting_fps = 1