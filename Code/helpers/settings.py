from dataclasses import dataclass
import numpy as np

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

    max_epochs=1000
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
    loss_weight_segmentation=0.000367
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