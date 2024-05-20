import sys
import os
sys.path.append(os.path.abspath('.'))

from dataclasses import dataclass
from helpers.classes import get_K_params
from data_management.DataManager import CALFData, collateGCN
import torch
import numpy as np
from helpers.classes import INVERSE_EVENT_DICTIONARY_V2_ALIVE as ann_encoder
import argparse
import pickle
from sklearn.linear_model import LogisticRegression


@dataclass
class Args:
    # DATA
    datapath="/project_antwerp/football_games"
    chunk_size = 60
    batch_size = 32
    input_channel = 13
    annotation_nr = 1
    receptive_field = 12
    fps = 5
    K_parameters = get_K_params(chunk_size)
    focused_annotation = "Shot"
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
    backbone_player = "GAT"
    load_weights=None
    model_name="Testing_Model"
    dim_capsule=16
    vocab_size=64
    pooling=None

    # SPOTTING MODULE
    sgementation_path = f"/project_antwerp/models/backbone_GAT.pth.tar",
    freeze_model = True
    spotting_fps = 1

model_paths = [f"/project_antwerp/models/finetuned/spotting_finetuned_{ann}.pth.tar" for ann in ann_encoder.values()]
ann_names = list(ann_encoder.values())
def main():
    args = Args
    for i, model_path in enumerate(model_paths):
        args.focused_annotation = ann_names[i]
        
        collate_fn = collateGCN
        calibration_dataset = CALFData(split="calibrate", args=args)

        calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                    batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
        model = torch.load(model_path)        
        predictions = []
        labels = []
        
        with torch.no_grad():  
            for _, targets, representations in calibration_loader:
                outputs = model(representations)  
                probabilities = torch.sigmoid(outputs)
                predictions.append(probabilities)
                labels.append(targets)
        
        predictions = torch.cat(predictions).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        predictions = predictions.reshape(-1,predictions.shape[2])
        labels = labels.reshape(-1,labels.shape[2])
        
        # Fit logistic regression as a calibrator
        calibrator = LogisticRegression(class_weight='balanced')
        calibrator.fit(predictions, labels)
        
        # iso_reg = IsotonicRegression(out_of_bounds='clip')
        # iso_reg.fit(predictions[:, class_index].reshape(-1, 1), binary_labels)  # Fit the calibrator
        # calibrated_probabilities[:, class_index] = iso_reg.transform(predictions[:, class_index].reshape(-1, 1))
        
        # Save the calibration models
        filename = f'/project_antwerp/calibrators/finetuned/{args.focused_annotation}_calibration_fine_tuned.pkl'
        pickle.dump(calibrator, open(filename, 'wb'))
            
        
if __name__ == '__main__':
    main()
