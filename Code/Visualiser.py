from DataManager import DataManager
from FileFinder import find_files
import numpy as np
import torch 
import copy
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import matplotlib.animation as animation
from mplsoccer.pitch import Pitch
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from sklearn.metrics import average_precision_score
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE as ann_encoder

class VisualiseDataset(Dataset):
    def __init__(self, args):
        
        listGames = find_files("../football_games")
        self.args = args

        DM = DataManager(files=listGames[0:1], framerate=args.framerate/25, alive=False)
        DM.read_games(ball_coords=True)
        DM.datasets[0].shape
        
        self.receptive_field = args.receptive_field*args.framerate
        self.window = args.chunk_size*args.framerate - self.receptive_field
        self.windows_count = np.floor(DM.datasets[0].shape[0]/self.window)-1

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
            x = torch.tensor(Features, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            self.representation.append(data)
    
    def __getitem__(self,index):
        indx = self.window*index
        clip_representation = copy.deepcopy(
            self.representation[indx:indx+self.window+self.receptive_field]
            )
        clip_annotation = copy.deepcopy(
            self.annotations[indx:indx+self.window+self.receptive_field]
            )
        return clip_representation, clip_annotation

    def __len__(self):
        return int(self.windows_count)

def collateVisGCN(list_of_examples):
    return Batch.from_data_list([x for b in list_of_examples for x in b[0]]),\
            np.stack([x[1] for x in list_of_examples], axis=0)

class Visualiser():
    def __init__(self, collate_fn, args, model, smoothing=False):   
        collate_fn = collate_fn
        data_visualise = VisualiseDataset(args=args)
        
        visualise_loader = torch.utils.data.DataLoader(data_visualise,
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        concatenated_seg =  np.zeros((int(data_visualise.receptive_field/2),args.annotation_nr))
        concatenated_spot =  torch.zeros((int(data_visualise.receptive_field/2),args.annotation_nr+2))
        # Set the columns to number of the classes
        annotations = np.zeros((int(data_visualise.receptive_field/2),args.annotation_nr))
        
        for representation, annotation in visualise_loader:
            segmentation, spotting = model(representation)
            
            segmentation = np.array([x[int(data_visualise.receptive_field/2):-int(data_visualise.receptive_field/2)] for x in segmentation.detach().numpy()])
            reshaped_seg = np.reshape(segmentation, (segmentation.shape[0]*segmentation.shape[1], segmentation.shape[2]))
            concatenated_seg = np.concatenate((concatenated_seg, reshaped_seg), axis=0)
            
            reshaped_spot = torch.reshape(spotting, (spotting.shape[0]*spotting.shape[1], spotting.shape[2]))
            concatenated_spot = torch.cat((concatenated_spot, reshaped_spot), dim=0)
            
            annotation = np.array([x[int(data_visualise.receptive_field/2):-int(data_visualise.receptive_field/2)] for x in annotation])
            reshaped_ann = np.reshape(annotation, (annotation.shape[0]*annotation.shape[1], annotation.shape[2]))
            annotations = np.concatenate((annotations, reshaped_ann), axis=0)

        if smoothing:
            smooth_seg = int(concatenated_seg.shape[0]/args.framerate)
            smooth_spot = int(concatenated_spot.shape[0]/args.framerate)
            smooth_ann = int(annotations.shape[0]/args.framerate)
            concatenated_seg = concatenated_seg.reshape((smooth_seg, args.framerate, args.annotation_nr)).mean(axis=1)
            concatenated_spot = concatenated_spot.reshape((smooth_spot, args.framerate, args.annotation_nr+2)).mean(axis=1)
            annotations = annotations.reshape((smooth_ann, args.framerate, args.annotation_nr))[:,0,:]

        self.segmentation = concatenated_seg
        self.spotting = concatenated_spot.detach().numpy()
        self.annotations = annotations
        self.args = args
        self.matrix = data_visualise.matrix
        self.ball_coords = data_visualise.ball_coords[0]
        self.smoothing = smoothing 

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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
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
        if self.smoothing:
            smooth_rate = 1/self.args.framerate

        ann_indx = ann_encoder[annotation]

        seg_pred = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),ann_indx], label='Predictions')
        seg_ann = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx], label='Annotations')
        ax2.set_title(f"Segmentation")
        ax2.legend()

        # Spotting plot
        spot_pred = ax3.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),2+ann_indx], label='Predictions')
        spot_ann = ax3.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx], label='Annotations')
        ax3.set_title(f"Spotting")
        ax3.legend()

        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            
            seg_pred[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),ann_indx])
            seg_ann[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx])
            
            spot_pred[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),2+ann_indx])
            spot_ann[0].set_data(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx])
            return (scat_home,scat_away,scat_ball)
        
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])
            
            if (self.smoothing) and (frame % self.args.framerate) == 0:
                seg_pred[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.segmentation[:int(frame*smooth_rate)+1, ann_indx])
                seg_ann[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.annotations[:int(frame*smooth_rate)+1, ann_indx])
            
                spot_pred[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.spotting[:int(frame*smooth_rate)+1, 2+ann_indx])
                spot_ann[0].set_data(np.arange(0, int(frame*smooth_rate) + 1), self.annotations[:int(frame*smooth_rate)+1, ann_indx])
            
            elif self.smoothing == False:
                seg_pred[0].set_data(np.arange(0, frame + 1), self.segmentation[:frame+1, ann_indx])
                seg_ann[0].set_data(np.arange(0, frame + 1), self.annotations[:frame+1, ann_indx])
                
                spot_pred[0].set_data(np.arange(0, frame + 1), self.spotting[:frame+1, 2+ann_indx])
                spot_ann[0].set_data(np.arange(0, frame + 1), self.annotations[:frame+1, ann_indx])
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
        if self.smoothing:
            smooth_rate = 1/self.args.framerate
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
        
        ann_indx = ann_encoder[annotation]

        # Draw segmentation plots 
        seg_pred = ax1.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.segmentation[:int(frame_threshold*smooth_rate),ann_indx], label='Prediction')
        seg_ann = ax1.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx], label='Annotations')
        ax1.set_title(f"Segmentation")
        ax1.legend()

        # Draw spotting plots
        spot_pred = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.spotting[:int(frame_threshold*smooth_rate),2+ann_indx], label='Prediction')
        spot_ann = ax2.plot(np.arange(0, int(frame_threshold*smooth_rate)), self.annotations[:int(frame_threshold*smooth_rate),ann_indx], label='Annotations')
        ax2.set_title(f"Spotting")
        ax2.legend()

        if save_dir != None:
            plt.savefig(save_dir)
        else:
            plt.show()
    
    def calculate_MAP(self):
        predictions = self.spotting[:,2:]
        if predictions.shape[0] != self.annotations.shape[0]:
            predictions = predictions[:self.annotations.shape[0],:]
        mAP = average_precision_score(self.annotations, predictions, average='macro')
        return mAP
    