import sys
import os
sys.path.append(os.path.abspath('.'))

from dataclasses import dataclass
from helpers.classes import get_K_params
from modules.GameAnalysis import GamaAnalysis
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE as ann_encoder
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from mplsoccer.pitch import Pitch
import argparse
from tqdm import tqdm
import time
import torch_geometric
import pickle

@dataclass
class Args:
    # DATA
    datapath="/project_antwerp/football_games"
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
    backbone_player = "GAT"
    load_weights=None
    model_name="Testing_Model"
    dim_capsule=16
    vocab_size=64
    pooling=None

    # SPOTTING MODULE
    sgementation_path = f"models/backbone_GAT.pth.tar"
    freeze_model = None
    spotting_fps = 1

def main():
    
    parser = argparse.ArgumentParser(prog="metric_visualiser", description="Visualise proposed metric")
    parser.add_argument("-m", "--model", help="The path to the base model")
    parser.add_argument("-a", "--annotation", help="Annotation to visualise top 10 missed predictions")
    parser.add_argument("-t", "--threshold", help="Number of clips of missed predictions")
    parser.add_argument("-o", "--output", help="The path to the folder to save visualisation")
    sys_args = parser.parse_args()
    
    # Generate predictions for given model
    # Selected game Brasil vs Belgium WC2018
    args = Args
    model = torch.load(sys_args.model)
    # model = torch.load("models/spotting_unfrozen_GCN.pth.tar")
    game_analyser = GamaAnalysis(args, model)
    results, annotations = game_analyser.predict_game(game_index=0, seg_model=False, calibrate=True, ann=None)

    # Select top 5 missed predictions
    predictions = game_analyser.spotting[:game_analyser.annotations.shape[0],:]
    prediction_diffrence = predictions - game_analyser.annotations

    ann_enc = ann_encoder[sys_args.annotation]
    missed_predictions_sorted = np.argsort(prediction_diffrence[:,ann_enc])
    top_missed_predictions =  select_indices(missed_predictions_sorted, int(sys_args.threshold))
    

    # Load the video
    video_path = '../football_games/BelgiumBrasil.mp4'
    capture = cv2.VideoCapture(video_path)
    kick_off = 208.3 # Manually set to synchronise video to predictions
    capture.set(cv2.CAP_PROP_POS_MSEC, kick_off * 1000)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if sys_args.threshold is None:
        sys_args.threshold = total_frames
    else:
        sys_args.threshold = int(sys_args.threshold)

    # Load tracking data
    pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)

    # get scalars to represent players position on the map
    scalars = (pitch.dim.pitch_length, pitch.dim.pitch_width)
    coords = game_analyser.matrix[:,0:2,:].copy()
    coords[:,0,:] = coords[:,0,:]*scalars[0]
    coords[:,1,:] = coords[:,1,:]*scalars[1]
    ball_coords = game_analyser.ball_coords.copy()
    ball_coords[:,0] = ball_coords[:,0]*scalars[0]
    ball_coords[:,1] = ball_coords[:,1]*scalars[1]
    # Details for scatter plots
    segmentation_fps = args.fps
    update_interval_scatter = int(video_fps / segmentation_fps) # Interval that scatter needs to be updated
    
    # Details about predictions
    prediction_fps = args.spotting_fps
    update_interval_prediction = int(video_fps / prediction_fps) # Interval that predictions need to be updated
    x_time = np.arange(game_analyser.spotting.shape[0]) / (prediction_fps*60)
    
    def generate_frames(frame, beginning_preds, save_dir=None):
        
        # INITIALISE FIGURE
        # Create the figure with a specified size
        fig = plt.figure(figsize=(20, 10))
        # Create a GridSpec layout
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # Allocate space for the video 
        video_ax = fig.add_subplot(gs[0:2, 0])
        video_ax.axis("off")
        ret, video_frame = capture.read()
        video_ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
        
        # Adjusted frame for scatter plot
        adjusted_frame_scatter = int(frame // update_interval_scatter - 20.5*update_interval_scatter) # Updated to synchronise with predictions
        beginning_scatter = int(np.max([adjusted_frame_scatter+1-1500, 0]))

        # Allocate space for the player tracking visualization
        player_tracking_ax = fig.add_subplot(gs[2, 0])
        player_tracking_ax.axis("off")
        pitch.draw(ax=player_tracking_ax)
        
        scat_home = player_tracking_ax.scatter(coords[adjusted_frame_scatter,0,:11].T, coords[adjusted_frame_scatter,1,:11].T, c="r", s=50)
        scat_away = player_tracking_ax.scatter(coords[adjusted_frame_scatter,0,11:].T, coords[adjusted_frame_scatter,1,11:].T, c="b", s=50)
        scat_ball = player_tracking_ax.scatter(ball_coords[adjusted_frame_scatter,0].T, ball_coords[adjusted_frame_scatter,1].T, c="black", s=50)
        scat_home.set_zorder(3) 
        scat_away.set_zorder(3)
        scat_ball.set_zorder(3)

        # Allocate space for the prediction distribution plots
        # We use a nested GridSpec for the 2x5 grid
        gs_right = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=gs[:, 1])

        # Create each of the 10 plots for prediction distributions
        adjusted_frame_preds = int(frame // update_interval_prediction)
        prediction_axes = []
        index = 0 
        for r in range(5):
            for c in range(2):
                ax = fig.add_subplot(gs_right[r, c])
                prediction_axes.append(ax)
                ax.set_title(f'{list(ann_encoder.keys())[index]}', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=8)
                ax.set_ylim(0,1)
                ax.set_facecolor('lightgray') 
                ax.grid(True, color='white')
                ax.plot(game_analyser.spotting[beginning_preds:adjusted_frame_preds+1, index])
                ax.plot(game_analyser.annotations[beginning_preds:adjusted_frame_preds+1, index])
                # ax.set_xlim(x_time[beginning], x_time[adjusted_frame+1])
                index += 1


        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.1)

        save_start = time.time()
        plt.savefig(f"{sys_args.output}/{save_dir}")
        save_end = time.time()
        save_time = save_end - save_start

        plt.close()
        return save_time
    
    # GENERATE FRAMES
    with tqdm(range(int(sys_args.threshold)), total=int(sys_args.threshold)) as t:        
        for clip in t:
            occurence_missed_preds = top_missed_predictions[clip]
            lower_bound_spot = -30*args.spotting_fps + occurence_missed_preds
            upper_bound_spot = 30*args.spotting_fps + occurence_missed_preds
            clip_range_spot = (lower_bound_spot, upper_bound_spot) 

            capture = cv2.VideoCapture(video_path)
            kick_off = 183.3 # Manually set to synchronise video to predictions
            kick_off_2nd = 3168
            end_half = 2975
            break_time = kick_off_2nd - end_half

            if lower_bound_spot <= 2760:
                capture.set(cv2.CAP_PROP_POS_MSEC, (kick_off+lower_bound_spot) * 1000) # As spotting is 1fps and we need miliseconds
            else:
                capture.set(cv2.CAP_PROP_POS_MSEC, (kick_off+break_time+lower_bound_spot) * 1000) # As spotting is 1fps and we need miliseconds

            # Chekck if paths exist
            if os.path.isdir(f"{sys_args.output}/{int(clip)}") is not True:
                os.mkdir(f"{sys_args.output}/{int(clip)}")

            for i, frame in enumerate(np.arange(start=clip_range_spot[0]*video_fps, stop=clip_range_spot[1]*video_fps)):
                save_dir =f"{int(clip)}/frame_animation{int(i)}.jpg"
                save_time = generate_frames(frame, lower_bound_spot, save_dir)
                desc = f"Frame {int(i)}: Saving time {save_time}"
                t.set_description(desc=desc)
    
            capture.release()
            
            # Save predictions
            with open(f"{sys_args.output}/{int(clip)}/predictions.pkl", 'wb') as f:
                # Use pickle.dump to save the arrays
                pickle.dump((
                    game_analyser.spotting[lower_bound_spot:upper_bound_spot], 
                    game_analyser.annotations[lower_bound_spot:upper_bound_spot]),f
                    )


def select_indices(indices, number):
    # List to hold the selected indices
    selected = []
    
    # Add the first index to the selected list
    selected.append(indices[0])

    # Iterate over the rest of the indices
    for index in indices[1:]:
        difference = [np.abs(i-index) for i in selected]
        # Check if index should be appended
        if np.any(np.array(difference)<60):
            continue
        else:
            selected.append(index)

        # Check to break the loop
        if len(selected) == number:
            break

    return selected

# Example usage
# ffmpeg -framerate 30 -i 'frame_animation%d.jpg' -c:v libx264 -pix_fmt yuv420p video.mp4
# rm *.jpg *.jpeg    

if __name__ == '__main__':
    main()


# args = Args
# collate_fn = collateVisGCN
# model_path = f"models/spotting_unfrozen_GCN.pth.tar"
# model = torch.load(model_path)
# visualiser = Visualiser(collate_fn, args, model, smooth_rate=None, val=True, seg_model=False)
# predictions = visualiser.spotting[:visualiser.annotations.shape[0],:]
# prediction_diffrence = predictions - visualiser.annotations
# ann_enc = ann_encoder["Duel"]
# top_10_missed_predictions = np.argsort(prediction_diffrence[:,ann_enc])
# top_10_missed_predictions = top_10_missed_predictions[:int(10)]