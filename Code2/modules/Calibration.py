import sys
import os
sys.path.append(os.path.abspath('.'))

from dataclasses import dataclass
from helpers.classes import get_K_params
from data_management.DataManager import CALFData, collateGCN
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from helpers.classes import INVERSE_EVENT_DICTIONARY_V2_ALIVE as ann_encoder
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from mplsoccer.pitch import Pitch
import argparse
from tqdm import tqdm
import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

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
    sgementation_path = f"models/spotting_unfrozen_GCN.pth.tar"
    freeze_model = None
    spotting_fps = 1

def main():

    parser = argparse.ArgumentParser(prog="metric_visualiser", description="Visualise proposed metric")
    parser.add_argument("-m", "--model", help="The path to the base model")
    sys_args = parser.parse_args()


    args = Args
    collate_fn = collateGCN
    calibration_dataset = CALFData(split="calibrate", args=args)

    calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model_path = sys_args.model
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

    calibrated_probabilities = np.zeros_like(predictions)
    class_index = 0
    for class_index in range(labels.shape[1]):  # Loop through each class
        # Extract the binary labels for the current class
        binary_labels = labels[:, class_index]
        
        # Fit logistic regression as a calibrator
        calibrator = LogisticRegression(class_weight='balanced')
        calibrator.fit(predictions[:, class_index].reshape(-1, 1), binary_labels)
        calibrated_probabilities[:, class_index] = calibrator.predict_proba(predictions[:, class_index].reshape(-1, 1))[:, 1]
        
        # iso_reg = IsotonicRegression(out_of_bounds='clip')
        # iso_reg.fit(predictions[:, class_index].reshape(-1, 1), binary_labels)  # Fit the calibrator
        # calibrated_probabilities[:, class_index] = iso_reg.transform(predictions[:, class_index].reshape(-1, 1))
        
        # Save the calibration models
        filename = f'calibrators/{ann_encoder[class_index]}_calibration.pkl'
        pickle.dump(calibrator, open(filename, 'wb'))
        
        
if __name__ == '__main__':
    main()
