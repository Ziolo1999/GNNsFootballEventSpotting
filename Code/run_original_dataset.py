from dataset import SoccerNetClips, collateGCN
import torch 

path ="../SoccerNetData/"

class Args():
    def  __init__(self):
        self.tiny=None
        self.features = "ResNET_TF2_PCA512.npy"
        self.max_epochs=1000
        self.load_weights=None
        self.model_name="calib_GCN"
        self.mode=0
        self.test_only=True
        self.challenge=True
        self.teacher=False
        self.class_split="visual"
        self.K_params=None
        self.num_features=512
        self.chunks_per_epoch=18000
        self.evaluation_frequency=20
        self.dim_capsule=16
        self.framerate=2
        self.chunk_size=120
        self.receptive_field=40
        self.lambda_coord=5.0
        self.lambda_noobj=0.5
        self.loss_weight_segmentation=0.000367
        self.loss_weight_detection=1.0
        self.num_detections=15
        self.feature_multiplier=2
        self.backbone_player="GCN"
        self.backbone_feature="2DConv"
        self.calibration=True
        self.calibration_field=True
        self.calibration_cone=True
        self.calibration_confidence=True
        self.dim_representation_w=64
        self.dim_representation_h=32
        self.dim_representation_c=3
        self.dim_representation_player=2
        self.dist_graph_player=25
        self.with_dropout=0.0
        self.batch_size=32
        self.LR=1e-03
        self.patience=25
        self.GPU=0 
        self.max_num_worker=1
        self.loglevel='INFO'
    
args = Args()


dataset = SoccerNetClips(path, args=args)

train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=2, shuffle=True, collate_fn=collateGCN)

data_iter = iter(train_loader)
next_batch = next(data_iter)
next_batch[3]