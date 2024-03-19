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

def collateVisGCN(list_of_examples):
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

class VisualiseDataset(Dataset):
    def __init__(self, args, val=True):
        
        listGames = find_files("../football_games")
        self.args = args

        if val:
            DM = DataManager(files=listGames[11:12], framerate=args.fps/25, alive=False)
        else:
            DM = DataManager(files=listGames[0:1], framerate=args.fps/25, alive=False)
        
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
            rows, cols = np.nonzero(DM.edges[0][frame])
            Edges = np.stack((rows, cols))
            edge_index = torch.tensor(Edges, dtype=torch.long)
            
            edge_attr = torch.tensor(
                    [
                        [
                        DM.edges[0][frame][x, y],
                        DM.velocity_diffs[0][frame][x, y],
                        DM.acceleration_diffs[0][frame][x, y],
                        DM.direction_diffs[0][frame][x, y]
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

class Visualiser():
    def __init__(self, collate_fn, args, model, smooth_rate=None, val=True, ann=None, seg_model=True):   
        
        collate_fn = collate_fn
        data_visualise = VisualiseDataset(args=args, val=val)
        
        visualise_loader = torch.utils.data.DataLoader(data_visualise,
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        if ann:
            concatenated_seg = np.ones((int(data_visualise.receptive_field/2), 1))
            concatenated_spot = np.zeros((int(data_visualise.receptive_field/2), 1))
            annotations = np.zeros((int(data_visualise.receptive_field/2),len(ann_encoder)))
        else:
            concatenated_seg = np.ones((int(data_visualise.receptive_field/2),args.annotation_nr))
            concatenated_spot = np.zeros((int(data_visualise.receptive_field/2),args.annotation_nr))
            annotations = np.zeros((int(data_visualise.receptive_field/2),args.annotation_nr))
        
        for representation, annotation in visualise_loader:
            model_output = model(representation)
            
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
                smoothed_ann = np.reshape(reshaped_ann, (-1, args.fps, reshaped_ann.shape[1])).max(axis=1)
                annotations = np.concatenate((annotations, smoothed_ann), axis=0)
                

        # if smooth_rate:
        #     smooth_seg = int(np.floor(concatenated_seg.shape[0]/smooth_rate))
        #     smooth_ann = int(np.floor(annotations.shape[0]/smooth_rate))
            
        #     concatenated_seg = concatenated_seg[:smooth_seg*smooth_rate].reshape((smooth_seg, smooth_rate, args.annotation_nr)).mean(axis=1)
        #     annotations = annotations[:smooth_ann*smooth_rate].reshape((smooth_ann, smooth_rate, args.annotation_nr)).max(axis=1)

        
        self.annotations = annotations
        self.spotting = concatenated_spot
        self.segmentation = 1-concatenated_seg
        self.segmentation = self.segmentation[:self.annotations.shape[0],:]
        
        self.args = args
        self.matrix = data_visualise.matrix
        self.ball_coords = data_visualise.ball_coords[0]
        self.smooth_rate = smooth_rate 
        self.seg_model = seg_model

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
        
        # Segmentation plot
        smooth_rate = 1
        if self.smooth_rate:
            smooth_rate = 1/self.smooth_rate

        ann_indx = ann_encoder[annotation]

        if self.seg_model:
            predictions = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),ann_indx], label='Predictions')
            ax2.set_title(f"Segmentation")
        else:
            predictions = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),ann_indx], label='Predictions')
            ax2.set_title(f"Spotting")

        seg_ann = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx], label='Annotations')
        ax2.legend()

        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            
            if self.seg_model:
                predictions[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),ann_indx])
            else:
                predictions[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),ann_indx])

            seg_ann[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx])
            
            return (scat_home,scat_away,scat_ball)
        
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])
            
            # if (self.smoothing) and (frame % self.args.fps) == 0:
            #     seg_pred[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.segmentation[:int(frame*smooth_rate)+1, ann_indx])
            #     seg_ann[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.annotations[:int(frame*smooth_rate)+1, ann_indx])
            
                # spot_pred[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.spotting[:int(frame*smooth_rate)+1, 2+ann_indx])
                # spot_ann[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.annotations[:int(frame*smooth_rate)+1, ann_indx])
            
            # elif self.smoothing == False:
            if self.seg_model:
                predictions[0].set_data(np.arange(0, frame + 1), self.segmentation[:frame+1, ann_indx])
            else:
                predictions[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, ann_indx])
            
            seg_ann[0].set_data(np.arange(0, frame + 1), self.annotations[:frame+1, ann_indx])
                
                # spot_pred[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, 2+ann_indx])
                # spot_ann[0].set_data(np.arange(0, frame + 1), self.annotations[:frame+1, ann_indx])
            # convert seconds to minutes and seconds
            # minutes, seconds = divmod(self.game_details[frame], 60)
            # format the output as mm:ss
            # formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
            # timestamp.set_text(f"Timestamp: {formatted_time}")
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
        smooth_rate = 1
        if self.smooth_rate:
            smooth_rate = 1/self.smooth_rate
        
        if annotation is None:
            cols = 2
            rows = int(np.ceil(self.args.annotation_nr)/2)
            fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 5*rows))
            
            # Draw segmentation plots 
            for i, ax in enumerate(axes.flatten()):
                ann = list(ann_encoder.keys())[i]
                seg_ann = ax.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),i], label='Annotations')
                
                if self.seg_model:
                    predictions = ax.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),i], label='Prediction')
                    ax.set_title(f"Segmentation {ann}")
                else:
                    predictions = ax.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),i], label='Prediction')
                    ax.set_title(f"Spotting {ann}")
                ax.legend()
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            ann_index = ann_encoder[annotation]
            axes.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_index], label='Annotations')
            
            if self.seg_model:
                axes.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),0], label='Prediction')
                axes.set_title(f"Segmentation {ann_index}")
            else:
                axes.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),0], label='Prediction')
                axes.set_title(f"Spotting {ann_index}")

            axes.legend()

        if save_dir != None:
            plt.savefig(save_dir)
        else:
            plt.show()
        return fig, axes
    
    def calculate_MAP(self):
        
        mAP_results = np.empty((self.annotations.shape[1]+1))
        
        if self.seg_model:
            predictions = self.segmentation
        else:
            predictions = self.spotting

        if predictions.shape[0] != self.annotations.shape[0]:
            predictions = predictions[:self.annotations.shape[0],:]
        
        total_mAP = average_precision_score(self.annotations, predictions, average='macro')
        mAP_results[0] = total_mAP

        for ann in range(self.annotations.shape[1]):
            gt = self.annotations[:, ann]
            pred = predictions[:, ann]
            mAP_score = average_precision_score(gt, pred, average='macro')
            mAP_results[ann+1] = mAP_score
        
        # col_names = ["Total"]+list(ann_encoder.keys())
        # mAP_df = pd.DataFrame(mAP_results, columns=col_names)
        return mAP_results
    
    def norm_evaluation_segmentation(self, ann=None):
        if ann:
            annotations = self.annotations[:, ann_encoder[ann]].reshape(-1,1)
        else:
            annotations = self.annotations

        precisions = np.zeros(annotations.shape[1])
        recalls = np.zeros(annotations.shape[1])
        f1_scores = np.zeros(annotations.shape[1])
        total_targets = np.zeros(annotations.shape)

        for ann in range(annotations.shape[1]):
            
            K_param = self.args.K_parameters[3,ann].item()/2*self.args.fps
            sigma = K_param/4
            scaler = K_param * 5/8 

            events = np.where(annotations[:,ann]==1)[0]

            frames = np.arange(0, annotations.shape[0])
            event_distribution = np.zeros((len(frames), len(events))) 

            # Calculate the maximum distribution value at each x
            for i, event in enumerate(events):
                event_distribution[:,i] = norm.pdf(frames, event, sigma) * scaler
            
            # Get final targets
            # event_distribution = np.concatenate((event_distribution, np.ones((event_distribution.shape[0], 1)) * 0.1), axis=1)
            target = np.max(event_distribution, axis=1) 
            total_targets[:,ann] = target

            # Get predictions
            predictions = self.segmentation[:annotations.shape[0], ann]
            
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


        return precisions, recalls, f1_scores, total_targets 
    
# from dataclasses import dataclass
# from helpers.classes import get_K_params
# @dataclass
# class Args:
#     receptive_field = 12
#     fps = 5
#     chunks_per_epoch = 1824
#     class_split = "alive"
#     chunk_size = 60
#     batch_size = 32
#     input_channel = 13
#     feature_multiplier=1
#     backbone_player = "GCN"
#     max_epochs=180
#     load_weights=None
#     model_name="Testing_Model"
#     dim_capsule=16
#     lambda_coord=5.0
#     lambda_noobj=0.5
#     patience=25
#     LR=1e-03
#     GPU=0 
#     max_num_worker=1
#     loglevel='INFO'
#     annotation_nr = 10
#     K_parameters = get_K_params(chunk_size)
#     focused_annotation = None
#     generate_augmented_data = True
#     sgementation_path = "models/gridsearch5.pth.tar"
#     freeze_model = True

# args = Args
# collate_fn = collateVisGCN
# model_path = "models/gridsearch5.pth.tar"
# model = torch.load(model_path)
# visualiser = Visualiser(collate_fn, args, model, smooth_rate=None, val=False)
# # precisions, recalls, f1_scores, total_targets = visualiser.norm_evaluation_segmentation()

# # plt.plot(total_targets[:, 8])
# # plt.plot(visualiser.segmentation[:,8])
# # plt.fill_between(np.arange(0,total_targets.shape[0]), np.minimum(total_targets[:, 8], visualiser.segmentation[:,8]), alpha=0.8)
# # plt.show()

# import cv2
# prediction_fps = 5

# video_path = '../football_games/BelgiumBrasil.mp4'
# capture = cv2.VideoCapture(video_path)
# kick_off = 3*60 + 21
# capture.set(cv2.CAP_PROP_POS_MSEC, kick_off * 1000)
# fps = capture.get(cv2.CAP_PROP_FPS)
# total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


# prediction_fps = 5
# frame_threshold = 1000
# update_interval_prediction = int(fps / prediction_fps)

# # create base animation
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# predictions = ax2.plot(np.arange(0, int(frame_threshold)), visualiser.annotations[:int(frame_threshold),8], label='Predictions')
# x_time = np.arange(visualiser.annotations.shape[0]) / (prediction_fps*60)

# ax2.set_title(f"Segmentation")
# # np.where(visualiser.annotations[:int(frame_threshold),8]==1)
# def init():
#     predictions[0].set_data(np.arange(0, int(frame_threshold)), visualiser.annotations[:int(frame_threshold),8])


# # get update function
# def update(frame):
#     ret, video_frame = capture.read()
#     if not ret:
#         print("Failed to grab frame.")
#         return
#     ax1.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))

#     adjusted_frame = frame // update_interval_prediction
#     beginning = int(np.max([adjusted_frame+1-1500, 0]))
#     predictions[0].set_data(x_time[beginning : adjusted_frame+1], visualiser.annotations[beginning:adjusted_frame+1, 8])
#     ax2.set_xlim(x_time[beginning], x_time[adjusted_frame+1])

# # use animation 
# ani = animation.FuncAnimation(fig=fig, func=update, frames=frame_threshold, init_func=init, interval=1)

# ani.save("animations/Predictions.mp4", writer='ffmpeg') 
# plt.show()
# capture.release()


# from data_management.DataPreprocessing import DatasetPreprocessor
# from data_management.FileFinder import find_files

# f = find_files("../football_games")[0]
# dataset = DatasetPreprocessor(1/5, f.name, alive=False)
# dataset._open_dataset(f.datafile, f.metafile, f.annotatedfile)

# # Generates node features
# player_violation = dataset._generate_node_features()

# # Generate edges and synchronise annotations
# dataset._generate_edges(threshold=None)
# dataset._synchronize_annotations(focused_annotation=None)


# dataset.animate_game(edge_threshold=None, direction=False, frame_threshold=1000, save_dir="animations/ShotSynchronisation.mp4", interval=200, annotation="Shot")
