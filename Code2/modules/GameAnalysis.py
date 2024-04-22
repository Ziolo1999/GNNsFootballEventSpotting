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
# from helpers.settings import Args

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
            for i, annotation_name in enumerate(ann_encoder.keys()):
                calibration_model_name = f"calibrators/{annotation_name}_calibration.pkl"
                calibration_model = pickle.load(open(calibration_model_name, 'rb'))
                self.spotting[:,i] = calibration_model.predict_proba(self.spotting[:,i].reshape(-1, 1))[:, 1]
        
        if seg_model:
            max_frame = int(np.min([self.segmentation.shape[0], self.annotations.shape[0]]))
            return self.segmentation[:max_frame,:], self.annotations[:max_frame,:]
        else:
            max_frame = int(np.min([self.spotting.shape[0], self.annotations.shape[0]]))
            return self.spotting[:max_frame,:], self.annotations[:max_frame,:]

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
                    ax.set_title(f"Segmentation {ann}")
                else:
                    predictions = ax.plot(np.arange(0, int(frame_threshold)), self.spotting[:int(frame_threshold),i], label='Prediction')
                    ax.set_title(f"Spotting {ann}")
                ax.legend()
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            ann_index = ann_encoder[annotation]
            axes.plot(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),ann_index], label='Annotations')
            
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
    
    def map_results(self):
        if self.seg_model:
            predictions = self.segmentation
        else:
            predictions = self.spotting
        
        mAP_results = calculate_MAP(self.annotations, predictions)
        return mAP_results
    
    def segmentation_evaluation(self, ann=None):
        if ann:
            annotations = self.annotations[:, ann_encoder[ann]].reshape(-1,1)
        else:
            annotations = self.annotations

        precisions, recalls, f1_scores = norm_evaluation_segmentation(annotations, self.segmentation, self.args, ann=ann)
        return precisions, recalls, f1_scores 
    
    def all_test_games_evaluation(self, args, last_game_index=2, seg_model=True, calibrate=False):
        all_results = np.empty((0, self.args.annotation_nr))
        all_annotations = np.empty((0, self.args.annotation_nr))

        for i in range(last_game_index):
            results, annotations = self.predict_game(game_index=i, seg_model=seg_model, calibrate=calibrate)
            all_results = np.concatenate((all_results, results), axis=0)
            all_annotations = np.concatenate((all_annotations, annotations), axis=0)
        
        if seg_model:
            return norm_evaluation_segmentation(all_annotations, all_results, args, ann=None)
        else:
            return calculate_MAP(all_annotations, all_results)



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

    for ann in range(annotations.shape[1]):
        gt = annotations[:, ann]
        pred = predictions[:, ann]
        mAP_score = average_precision_score(gt, pred, average='macro')
        mAP_results[ann+1] = mAP_score
    
    return mAP_results

def norm_evaluation_segmentation(annotations, segmentation, args, ann=None):
    precisions = np.zeros(annotations.shape[1])
    recalls = np.zeros(annotations.shape[1])
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

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        precisions[ann] = precision
        recalls[ann] = recall
        f1_scores[ann] = f1_score


    return precisions, recalls, f1_scores 
# calibration_model_name = f"calibrators/Duel_calibration.pkl"
# calibration_model = pickle.load(open(calibration_model_name, 'rb'))
# calibration_model
# self.spotting[:,i] = calibration_model.predict_proba(self.spotting[:,i].reshape(-1, 1))[:, 1]
