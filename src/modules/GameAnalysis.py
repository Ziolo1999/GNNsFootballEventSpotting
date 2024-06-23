from data_management.DataManager import DataManager
from data_management.FileFinder import find_files
import numpy as np
import torch 
import copy
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from sklearn.metrics import average_precision_score
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE as ann_encoder
import pandas as pd
import torch.nn.functional as F
from scipy.stats import norm
import pickle
from tqdm import tqdm
from sklearn.metrics import auc

class AnalysisDataset(Dataset):
    def __init__(self, args, game_index=0):
        
        listGames = find_files("../football_games")
        self.args = args

        DM = DataManager(files=listGames[game_index:game_index+1], framerate=args.fps/25, alive=False)
        
        DM.read_games(ball_coords=True)
        DM.datasets[0].shape
        
        self.receptive_field = int(args.receptive_field*args.fps)
        self.window = int(args.chunk_size*args.fps - self.receptive_field)
        self.chunk =  int(args.chunk_size * args.fps)
        self.chunk_counts = int(np.floor(
            (DM.datasets[0].shape[0]-self.chunk)/self.window))
        
        # self.full_receptive_field = self.chunk - self.window

        self.representation = []
        self.matrix = DM.datasets[0]
        self.ball_coords = DM.ball_coords
        self.annotations = DM.annotations[0]

        for frame in range(DM.datasets[0].shape[0]):
            # Get nodes features
            Features = DM.datasets[0][frame].T
            # Get edges indicses
            rows, cols = np.nonzero(DM.edge_weights[0][frame])
            Edges = np.stack((rows, cols))
            edge_index = torch.tensor(Edges, dtype=torch.long)
            edge_attr = torch.tensor(
                [
                    [
                    DM.distances[0][frame][x, y],
                    DM.velocity_diffs[0][frame][x, y],
                    DM.acceleration_diffs[0][frame][x, y],
                    DM.direction_diffs[0][frame][x, y],
                    DM.edge_weights[0][frame][x, y]
                    ] for x, y in zip(rows, cols)
                ], 
                dtype=torch.float
            )
            
            x = torch.tensor(Features, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            self.representation.append(data)
    
    def __getitem__(self,index):
        indx = int(self.window * index)
        clip_representation = copy.deepcopy(
            self.representation[indx:indx+self.chunk]
            )
        clip_annotation = copy.deepcopy(
            self.annotations[indx:indx+self.chunk]
            )
        return clip_representation, clip_annotation

    def __len__(self):
        return int(self.chunk_counts)

class GamaAnalysis():
    def __init__(self, args, model):   
        
        self.collate_fn = collateAnalysis
        self.model = model.eval()
        self.args = args

    
    def predict_game(self, game_index, seg_model=True, calibrate=False, ann=None): 
        data_visualise = AnalysisDataset(args=self.args, game_index=game_index)
        
        visualise_loader = torch.utils.data.DataLoader(data_visualise,
                            batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)
        # GENERATE EMPTY ARRAYS
        if ann:
            concatenated_seg = np.ones((int(data_visualise.receptive_field/2), 1))
            concatenated_spot = np.zeros((int(data_visualise.receptive_field/2), 1))
            annotations = np.zeros((int(data_visualise.receptive_field/2),len(ann_encoder)))
        else:
            concatenated_seg = np.ones((int(data_visualise.receptive_field/2),self.args.annotation_nr))
            concatenated_spot = np.zeros((int(data_visualise.receptive_field/2),self.args.annotation_nr))
            annotations = np.zeros((int(data_visualise.receptive_field/2),self.args.annotation_nr))
        
        # GENERATE ALL PREDICTIONS
        for representation, annotation in visualise_loader:
            model_output = self.model(representation)
            
            if seg_model:
                segmentation = np.array([x[int(data_visualise.receptive_field/2):-int(data_visualise.receptive_field/2)] for x in model_output.detach().numpy()])
                reshaped_seg = np.reshape(segmentation, (segmentation.shape[0]*segmentation.shape[1], segmentation.shape[2]))
                concatenated_seg = np.concatenate((concatenated_seg, reshaped_seg), axis=0)

                annotation = np.array([x[int(data_visualise.receptive_field/2):-int(data_visualise.receptive_field/2)] for x in annotation])
                reshaped_ann = np.reshape(annotation, (annotation.shape[0]*annotation.shape[1], annotation.shape[2]))
                annotations = np.concatenate((annotations, reshaped_ann), axis=0)
            else:
                model_output = F.sigmoid(model_output)
                spotting = np.array([x for x in model_output.detach().numpy()])
                reshaped_spot = np.reshape(spotting, (spotting.shape[0]*spotting.shape[1], spotting.shape[2]))
                concatenated_spot = np.concatenate((concatenated_spot, reshaped_spot), axis=0)

                annotation = np.array([x[int(data_visualise.receptive_field/2):-int(data_visualise.receptive_field/2)] for x in annotation])
                reshaped_ann = np.reshape(annotation, (annotation.shape[0]*annotation.shape[1], annotation.shape[2]))
                smoothed_ann = np.reshape(reshaped_ann, (-1, self.args.fps, reshaped_ann.shape[1])).max(axis=1)
                annotations = np.concatenate((annotations, smoothed_ann), axis=0)
        
        self.annotations = annotations
        self.spotting = concatenated_spot
        
        if self.args.generate_artificial_targets is not True:
            self.segmentation = 1-concatenated_seg
        else:
            self.segmentation = concatenated_seg
        self.segmentation = self.segmentation[:self.annotations.shape[0],:]
        
        self.matrix = data_visualise.matrix
        self.ball_coords = data_visualise.ball_coords[0]
        self.seg_model = seg_model

        # # CALIBRATION STEP
        if calibrate:
            if self.args.annotation_nr==1:
                calibration_type = "fine_tuned"
                calibration_model_name = f"calibrators/{calibration_type}/{self.args.focused_annotation}_calibration_{calibration_type}.pkl"
                calibration_model = pickle.load(open(calibration_model_name, 'rb'))
                self.spotting[:,0] = calibration_model.predict_proba(self.spotting[:,0].reshape(-1, 1))[:, 1]
            else:
                if self.args.pooling is None:
                    calibration_type = self.args.backbone_player
                else:
                    calibration_type = "NetVLAD"

                for i, annotation_name in enumerate(ann_encoder.keys()):
                    calibration_model_name = f"calibrators/{calibration_type}/{annotation_name}_calibration_{calibration_type}.pkl"
                    calibration_model = pickle.load(open(calibration_model_name, 'rb'))
                    self.spotting[:,i] = calibration_model.predict_proba(self.spotting[:,i].reshape(-1, 1))[:, 1]
        
        if seg_model:
            max_frame = int(np.min([self.segmentation.shape[0], self.annotations.shape[0]]))
            self.segmentation = self.segmentation[:max_frame,:]
            self.annotations = self.annotations[:max_frame,:]
            return self.segmentation, self.annotations
        else:
            max_frame = int(np.min([self.spotting.shape[0], self.annotations.shape[0]]))
            self.spotting = self.spotting[:max_frame,:]
            self.annotations = self.annotations[:max_frame,:]
            return self.spotting, self.annotations

    def visualize(self, frame_threshold=None, save_dir=None, interval=1, annotation="Shot"):
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
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

        ann_indx = ann_encoder[annotation]

        if self.seg_model:
            predictions = ax2.plot(np.arange(0, int(frame_threshold)), self.segmentation[:int(frame_threshold),ann_indx], label='Predictions')
            ax2.set_title(f"Segmentation")
        else:
            predictions = ax2.plot(np.arange(0, int(frame_threshold)), self.spotting[:int(frame_threshold),ann_indx], label='Predictions')
            ax2.set_title(f"Spotting")

        seg_ann = ax2.plot(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),ann_indx], label='Annotations')
        ax2.legend()

        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            
            if self.seg_model:
                predictions[0].set_data(np.arange(0, int(frame_threshold)), self.segmentation[:int(frame_threshold),ann_indx])
            else:
                predictions[0].set_data(np.arange(0, int(frame_threshold)), self.spotting[:int(frame_threshold),ann_indx])

            seg_ann[0].set_data(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),ann_indx])
            
            return (scat_home,scat_away,scat_ball)
        
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])

            if self.seg_model:
                predictions[0].set_data(np.arange(0, frame + 1), self.segmentation[:frame+1, ann_indx])
            else:
                predictions[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, ann_indx])
            
            seg_ann[0].set_data(np.arange(0, frame + 1), self.annotations[:frame+1, ann_indx])
    
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

    def plot_predictions(self, frame_threshold=None, save_dir=None, annotation="Shot"):
        # Set the smoothing rate
        if annotation is None:
            cols = 2
            rows = int(np.ceil(self.args.annotation_nr)/2)
            fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 5*rows))
            
            # Draw segmentation plots 
            for i, ax in enumerate(axes.flatten()):
                ann = list(ann_encoder.keys())[i]
                seg_ann = ax.plot(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),i], label='Annotations')
                
                if self.seg_model:
                    predictions = ax.plot(np.arange(0, int(frame_threshold)), self.segmentation[:int(frame_threshold),i], label='Prediction')
                    ax.set_title(f"Segmentation {ann}", fontsize=15)
                else:
                    predictions = ax.plot(np.arange(0, int(frame_threshold)), self.spotting[:int(frame_threshold),i], label='Prediction')
                    ax.set_title(f"Spotting {ann}", fontsize=15)
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.legend(fontsize=15)
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            ann_index = ann_encoder[annotation]
            axes.plot(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),ann_index], label='Annotations')
            axes.tick_params(axis='x', labelsize=15)
            axes.tick_params(axis='y', labelsize=15)
            
            if self.seg_model:
                axes.plot(np.arange(0, int(frame_threshold)), self.segmentation[:int(frame_threshold),0], label='Prediction')
                axes.set_title(f"Segmentation {ann_index}")
    
            else:
                axes.plot(np.arange(0, int(frame_threshold)), self.spotting[:int(frame_threshold),0], label='Prediction')
                axes.set_title(f"Spotting {ann_index}")
                

            axes.legend()

        if save_dir != None:
            plt.savefig(save_dir)
        else:
            plt.show()
        return fig, axes
    
    def map_results(self, ann):
        predictions = self.spotting
        
        if ann:
            annotations = self.annotations[:, ann_encoder[ann]].reshape(-1,1)
        else:
            annotations = self.annotations   

        mAP_results = calculate_MAP(annotations, predictions)
        return mAP_results
    
    def segmentation_evaluation(self, ann=None):
        if ann:
            annotations = self.annotations[:, ann_encoder[ann]].reshape(-1,1)
        else:
            annotations = self.annotations

        precisions, recalls, f1_scores = norm_evaluation_segmentation(annotations, self.segmentation, self.args, ann=ann)
        return precisions, recalls, f1_scores 
    
    def precision_recall_f1(self, pred_threshold=0.6, error_margin=0):
        return precision_recall_f1_score(self.spotting, self.annotations, pred_threshold=pred_threshold, error_margin=error_margin)
    
    def ROC_AUC(self,error_margin=0):
        return draw_ROC_curve(self.spotting, self.annotations, error_margin=error_margin)
    
    def all_test_games_evaluation(self, args, last_game_index=2, seg_model=True, calibrate=False, ann=None, type_eval="segmentation_evaluation"):
        if ann:
            all_results = np.empty((0, 1))
            all_annotations = np.empty((0, 1))
        else:
            all_results = np.empty((0, self.args.annotation_nr))
            all_annotations = np.empty((0, self.args.annotation_nr))

        for i in range(last_game_index):
            results, annotations = self.predict_game(game_index=i, seg_model=seg_model, calibrate=calibrate, ann=ann)
            if ann:
                annotations = annotations[:, ann_encoder[ann]].reshape(-1,1)

            all_results = np.concatenate((all_results, results), axis=0)
            all_annotations = np.concatenate((all_annotations, annotations), axis=0)
        
        if type_eval=="segmentation_evaluation":
            return norm_evaluation_segmentation(all_annotations, all_results, args, ann=ann)
        elif type_eval=="map_evaluation":
            return calculate_MAP(all_annotations, all_results)
        elif type_eval=="clip_based_precision_recall":
            return precision_recall_f1_score(all_results, all_annotations, pred_threshold=0.6, error_margin=0)
        elif type_eval=="clip_based_ROC_AUC":
            return draw_ROC_curve(all_results, all_annotations, error_margin=0)
        elif type_eval=="precision_recall_curve":
            return draw_prec_rec_curve(all_results, all_annotations, error_margin=0, plot=False)

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
    
    def precision_recall_f1(self, pred_threshold=0.6, error_margin=1):
        return precision_recall_f1_score(self.all_targets, self.all_outputs, pred_threshold=pred_threshold, error_margin=error_margin)
    
    def ROC_AUC(self,error_margin=1):
        return draw_ROC_curve(self.all_targets, self.all_outputs, error_margin=error_margin)

def collateAnalysis(list_of_examples):
    return Batch.from_data_list([x for b in list_of_examples for x in b[0]]),\
            np.stack([x[1] for x in list_of_examples], axis=0)

def average_segmentation(segmentation_results, window):
    x_axis = int(segmentation_results.shape[0])
    y_axis = ((segmentation_results.shape[0]-1) * window + segmentation_results.shape[1]) 
    z_axis = (segmentation_results.shape[2])
    exceeded_segmentation = np.empty((x_axis, y_axis, z_axis))
    
    for index, batch in enumerate(segmentation_results):
        exceeded_segmentation[index, index*window:index*window+segmentation_results.shape[1], :] = batch
    
    avg_segmentation = np.mean(exceeded_segmentation, axis=0, where=(exceeded_segmentation != 0))
    return avg_segmentation    

def calculate_MAP(annotations, predictions):
        
    mAP_results = np.empty((annotations.shape[1]+1))
    total_mAP = average_precision_score(annotations, predictions, average='macro')
    mAP_results[0] = total_mAP
    
    if annotations.shape[1]>1:
        for ann in range(annotations.shape[1]):
            gt = annotations[:, ann]
            pred = predictions[:, ann]
            mAP_score = average_precision_score(gt, pred, average='macro')
            mAP_results[ann+1] = mAP_score
        return mAP_results
    else:
        return total_mAP    
    

def norm_evaluation_segmentation(annotations, segmentation, args, ann=None):
    PORs = np.zeros(annotations.shape[1])
    ACRs = np.zeros(annotations.shape[1])
    f1_scores = np.zeros(annotations.shape[1])
    total_targets = np.zeros(annotations.shape)

    for ann in range(annotations.shape[1]):
        
        K_param = args.K_parameters[3,ann].item()/2*args.fps
        sigma = K_param/4
        scaler = K_param * 5/8 

        events = np.where(annotations[:,ann]==1)[0]

        frames = np.arange(0, annotations.shape[0])
        event_distribution = np.zeros((len(frames), len(events))) 

        # Calculate the maximum distribution value at each x
        for i, event in enumerate(events):
            event_distribution[:,i] = norm.pdf(frames, event, sigma) * scaler
        
        # Get final targets
        target = np.max(event_distribution, axis=1) 
        total_targets[:,ann] = target

        # Get predictions
        predictions = segmentation[:annotations.shape[0], ann]
        
        # Get evaluation metrics
        TP = np.sum(np.minimum(predictions, target))  # Overlap
        FP = np.sum(predictions - np.minimum(predictions, target))  # Excess in predicted
        FN = np.sum(target - np.minimum(predictions, target))  # Shortfall in actual

        POR = TP / (TP + FP)
        ACR = TP / (TP + FN)
        f1_score = 2 * (POR * ACR) / (POR + ACR)

        PORs[ann] = POR
        ACRs[ann] = ACR
        f1_scores[ann] = f1_score


    return PORs, ACRs, f1_scores 

def precision_recall_f1_score(results, annotations, pred_threshold=0.6, error_margin=0):
    TP, TN, FN, FP = get_event_evaluation(results, annotations, pred_threshold=pred_threshold, error_margin=error_margin)
    
    p = np.array(TP)/(np.array(TP) + np.array(FP))
    r = np.array(TP)/(np.array(TP) + np.array(FN))
    f1_score = 2*p*r/(p+r)

    return p, r, f1_score

def draw_ROC_curve(results, annotations, error_margin=1):
    TPRs = np.empty((0, results.shape[1]))
    FPRs = np.empty((0, results.shape[1]))

    for threshold in range(0, 105, 1):
        threshold/=100
        TP, TN, FN, FP = get_event_evaluation(results, annotations, pred_threshold=threshold, error_margin=error_margin)
        TPR = np.expand_dims(np.array(TP)/(np.array(TP) + np.array(FN)), 0)
        FPR = np.expand_dims(np.array(FP)/(np.array(TN) + np.array(FP)), 0)
        TPRs = np.concatenate((TPRs,TPR), axis=0)
        FPRs = np.concatenate((FPRs,FPR), axis=0)
    
    cols = 2
    rows = int(np.ceil(results.shape[1])/2)
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 5*rows))

    for i, ax in enumerate(axes.flatten()):
        ann = list(ann_encoder.keys())[i]
        ax.plot(FPRs[:,i], TPRs[:, i])
        ax.set_title(f"ROC curve: {ann}")
        ax.set_facecolor('lightgray') 
        ax.grid(True, color='white')
        ax.fill_between(FPRs[:,i], TPRs[:, i], alpha=0.5)
    
    AUC = [auc(FPRs[:,i], TPRs[:,i]) for i in range(FPRs.shape[1])]
    return AUC

def draw_prec_rec_curve(results, annotations, error_margin=0, plot=False):
    precisions = []
    recalls = []
    
    for threshold in range(0, 105, 5):
        threshold/=100
        TP, TN, FN, FP = get_event_evaluation(results, annotations, pred_threshold=threshold, error_margin=error_margin)
        TP = np.sum(TP)
        FN = np.sum(FN)
        FP = np.sum(FP)

        p = np.array(TP)/(np.array(TP) + np.array(FP))
        r = np.array(TP)/(np.array(TP) + np.array(FN))
        
        precisions.append(p)
        recalls.append(r)
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(precisions, recalls)
        ax.set_title(f"Precision-Recall curve")
        ax.set_facecolor('lightgray') 
        ax.grid(True, color='white')
        ax.fill_between(precisions, recalls, alpha=0.5)
    
    return precisions, recalls

def interpolate_matrix(data):
    for ann in range(data.shape[1]):
        for i in range(1, data.shape[0]-1): 
            if (data[i - 1, ann] == 1) and (data[i + 1, ann] == 1):
                data[i, ann] = 1
    return data

def find_events_in_column(column):
    # Calculate differences between consecutive elements in the column
    diffs = np.diff(column)

    # Starts of patterns are where the diff is 1 (0 to 1 transition)
    starts = np.where(diffs == 1)[0] + 1
    
    # Ends of patterns are where the diff is -1 (1 to 0 transition)
    ends = np.where(diffs == -1)[0]
    
    # If column starts with 1, prepend the first index
    if column[0] == 1:
        starts = np.insert(starts, 0, 0)
    
    # If column ends with 1, append the last index
    if column[-1] == 1:
        ends = np.append(ends, len(column) - 1)
    
    return starts, ends

def find_negative_events_in_column(column, min_length=5):
    # Calculate differences between consecutive elements in the column
    diffs = np.diff(column)

    # Starts of zero patterns are where the diff is -1 (1 to 0 transition)
    starts = np.where(diffs == -1)[0] + 1
    
    # Ends of zero patterns are where the diff is 1 (0 to 1 transition)
    ends = np.where(diffs == 1)[0]
    
    # If column starts with 0, prepend the first index
    if column[0] == 0:
        starts = np.insert(starts, 0, 0)
    
    # If column ends with 0, append the last index
    if column[-1] == 0:
        ends = np.append(ends, len(column) - 1)
    
    # Filter divide events to include only those that are at least `min_length` long
    start_list = []
    end_list = []
    for start, end in zip(starts, ends):
        length = end - start + 1
        if length >= min_length:
            event_count_ceil = int(np.ceil(length/min_length))
            event_count_floor = int(np.floor(length/min_length))
            for i in range(event_count_ceil):
                start_list.append(start+i*min_length)
                max_end = np.max([0,(event_count_floor-i)*min_length-1])
                end_list.append(end-max_end)

    
    return np.array(start_list), np.array(end_list)

def find_events_matrix(matrix, type_event=1):
    # Initialize lists to hold start and end indices for all columns
    all_starts = []
    all_ends = []
    
    # Iterate over each column in the matrix
    for i in range(matrix.shape[1]): 
        column = matrix[:, i]  
        if type_event == 1:
            starts, ends = find_events_in_column(column)
        else:
            starts, ends = find_negative_events_in_column(column)
        all_starts.append(starts)
        all_ends.append(ends)
    
    return all_starts, all_ends

def get_event_evaluation(results, annotations, pred_threshold=0.6, error_margin=1):

    # Interpolation of the annotations
    # annotations_interpolated = interpolate_matrix(annotations)
    annotations_interpolated = annotations
    # Get binary predictions
    results_binary = np.where((results>pred_threshold), 1,0)
    results_interpolated = results_binary
    # Interpolate results
    # results_interpolated = interpolate_matrix(results_binary)

    # Storage for eval metrics
    TP = [0 for i in range(annotations.shape[1])]
    TN = [0 for i in range(annotations.shape[1])]
    FN = [0 for i in range(annotations.shape[1])]
    FP = [0 for i in range(annotations.shape[1])]

    # Update TP and TN
    # Find events in annotations
    starts, ends = find_events_matrix(annotations_interpolated)
    for i, (start, end) in enumerate(zip(starts, ends)):
        for (s, e) in zip(start, end):
            adjusted_start = s-error_margin
            adjusted_end = e+1+error_margin
            # Get events
            event = results_interpolated[adjusted_start:adjusted_end, i]
            diff = np.ones(len(event)) - event
            if np.any(diff==0):
                TP[i] += 1
            elif np.all(diff==1):
                FN[i] += 1

    # Update FP and TN
    # Find events in results
    starts, ends = find_events_matrix(annotations_interpolated, type_event=0)
    for i, (start, end) in enumerate(zip(starts, ends)):
        for (s, e) in zip(start, end):
            adjusted_start = s+error_margin
            adjusted_end = e+1-error_margin
            # Get events
            event = results_interpolated[adjusted_start:adjusted_end, i]
            diff = np.zeros(len(event)) - event
            if np.all(diff==0):
                TN[i] += 1
            elif np.any(diff==-1):
                FP[i] += 1

    return TP, TN, FN, FP
