import torch
import torch.nn.functional as F
from dataclasses import dataclass
from helpers.settings import get_K_params
from data_management.DataManager import CALFData, collateGCN
from tqdm import tqdm
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE as ann_encoder
import pickle
import numpy as np
from sklearn.metrics import average_precision_score

# empty = torch.empty((0,10))
# x = torch.rand((10,10))
# x
# res = torch.cat((empty, x), dim=0)
# res.reshape(-1)


# @dataclass
# class Args:
#     # DATA
#     chunk_size = 60
#     batch_size = 32
#     input_channel = 13
#     annotation_nr = 10
#     receptive_field = 12
#     fps = 5
#     K_parameters = get_K_params(chunk_size)
#     focused_annotation = None
#     generate_augmented_data = True
#     class_split = "alive"
#     generate_artificial_targets = False
    
#     # TRAINING
#     chunks_per_epoch = 1824
#     lambda_coord=5.0
#     lambda_noobj=0.5
#     patience=25
#     LR=1e-03
#     max_epochs=180
#     GPU=0 
#     max_num_worker=1
#     loglevel='INFO'
    
#     # SEGMENTATION MODULE
#     feature_multiplier=1
#     backbone_player = "GCN"
#     load_weights=None
#     model_name="Testing_Model"
#     dim_capsule=16
#     vocab_size=64
#     pooling=None

#     # SPOTTING MODULE
#     sgementation_path = f"models/spotting_unfrozen_GCN.pth.tar"
#     freeze_model = None
#     spotting_fps = 1

# args = Args
# collate_fn = collateGCN
# validation_dataset = CALFData(split="validate", args=args)
# validate_loader = torch.utils.data.DataLoader(validation_dataset,
#             batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# _, targets, representations = next(iter(validate_loader))
# representations
# targets.shape

# model_name =f"models/spotting_unfrozen_GCN.pth.tar"
# model = torch.load(model_name)
# res = model(representations)
# len(validate_loader)

class Evaluator():
    def __init__(self, args, model, test_loader, calibrate=True):
        self.args = args
        self.model = model
        self.test_loader = test_loader

        self.all_targets = torch.empty((0,self.args.annotation_nr))
        self.all_outputs = torch.empty((0,self.args.annotation_nr))
        
        epochs = self.args.max_epochs
        # Generate predictions
        with tqdm(range(epochs), total=epochs) as t:
            desc = "Generating predictions for evaluation:"
            t.set_description(desc=desc)

            for _ in t:
                for _, targets, representations in self.test_loader:
                    outputs = self.model(representations)
                    outputs = F.sigmoid(outputs)
                    outputs = outputs.reshape(-1,self.args.annotation_nr)
                    self.all_outputs = torch.cat((self.all_outputs, outputs), dim=0)
                    
                    targets = targets.reshape(-1,self.args.annotation_nr)
                    self.all_targets = torch.cat((self.all_targets, targets), dim=0)
        # To numpy 
        self.all_outputs = self.all_outputs.detach().numpy()
        self.all_targets = self.all_targets.detach().numpy()

        # Calibrate results
        if calibrate:
            for i, annotation_name in enumerate(ann_encoder.keys()):
                calibration_model_name = f"calibrators/{annotation_name}_calibration.pkl"
                calibration_model = pickle.load(open(calibration_model_name, 'rb'))
                self.all_outputs[:,i] = calibration_model.predict_proba(self.all_outputs[:,i].reshape(-1, 1))[:, 1]
    
    def calculate_map(self):
        
        mAP_results = np.empty((self.all_outputs.shape[1]+1))
        total_mAP = average_precision_score(self.all_targets, self.all_outputs, average='macro')

        mAP_results[0] = total_mAP

        for ann in range(self.all_targets.shape[1]):
            gt = self.all_targets[:, ann]
            pred = self.all_outputs[:, ann]
            mAP_score = average_precision_score(gt, pred, average='macro')
            mAP_results[ann+1] = mAP_score
        
        return mAP_results





