import sys
import os
sys.path.append(os.path.abspath('.'))

from dataclasses import dataclass
from helpers.classes import get_K_params
from modules.Visualiser import collateVisGCN, Visualiser
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

@dataclass
class Args:
    receptive_field = 12
    fps = 5
    chunks_per_epoch = 1824
    class_split = "alive"
    chunk_size = 60
    batch_size = 32
    input_channel = 13
    feature_multiplier=1
    backbone_player = "GCN"
    max_epochs=180
    load_weights=None
    model_name="Testing_Model"
    dim_capsule=16
    lambda_coord=5.0
    lambda_noobj=0.5
    patience=25
    LR=1e-03
    GPU=0 
    max_num_worker=1
    loglevel='INFO'
    annotation_nr = 10
    K_parameters = get_K_params(chunk_size)
    focused_annotation = None
    generate_augmented_data = True
    sgementation_path = "models/gridsearch5.pth.tar"
    freeze_model = True

def main():
    
    parser = argparse.ArgumentParser(prog="metric_visualiser", description="Visualise proposed metric")
    parser.add_argument("-m", "--model", help="The path to the base model")
    parser.add_argument("-t", "--threshold", help="The number of frames to visualise")
    parser.add_argument("-o", "--output", help="The path to the folder to save visualisation")
    sys_args = parser.parse_args()
    
    # Generate predictions for given model
    # Selected game Brasil vs Belgium WC2018
    args = Args
    collate_fn = collateVisGCN
    model = torch.load(sys_args.model)
    visualiser = Visualiser(collate_fn, args, model, smooth_rate=None, val=False)

    # Load the video
    video_path = '../football_games/BelgiumBrasil.mp4'
    capture = cv2.VideoCapture(video_path)
    kick_off = 208.3 # Manually set to synchronise video to predictions
    capture.set(cv2.CAP_PROP_POS_MSEC, kick_off * 1000)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if sys_args.threshold is None:
        sys_args.threshold = total_frames
    else:
        sys_args.threshold = int(sys_args.threshold)

    # Load tracking data
    pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)

    # get scalars to represent players position on the map
    scalars = (pitch.dim.pitch_length, pitch.dim.pitch_width)
    coords = visualiser.matrix.copy()
    coords[:,0,:] = coords[:,0,:]*scalars[0]
    coords[:,1,:] = coords[:,1,:]*scalars[1]
    ball_coords = visualiser.ball_coords.copy()
    ball_coords[:,0] = ball_coords[:,0]*scalars[0]
    ball_coords[:,1] = ball_coords[:,1]*scalars[1]
    
    # Details about predictions
    prediction_fps = args.fps
    update_interval_prediction = int(fps / prediction_fps) # Interval that predictions need to be updated
    x_time = np.arange(visualiser.segmentation.shape[0]) / (prediction_fps*60)

    # INITIALISE FIGURE
    # Create the figure with a specified size
    fig = plt.figure(figsize=(20, 10))
    # Create a GridSpec layout
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Allocate space for the video 
    video_ax = fig.add_subplot(gs[0:2, 0])
    video_ax.axis("off")
    
    # Allocate space for the player tracking visualization
    player_tracking_ax = fig.add_subplot(gs[2, 0])
    player_tracking_ax.axis("off")
    pitch.draw(ax=player_tracking_ax)

    scat_home = player_tracking_ax.scatter([], [], c="r", s=50)
    scat_away = player_tracking_ax.scatter([], [], c="b", s=50)
    scat_ball = player_tracking_ax.scatter([], [], c="black", s=50)
    scat_home.set_zorder(3) 
    scat_away.set_zorder(3)
    scat_ball.set_zorder(3)

    # Allocate space for the prediction distribution plots
    # We use a nested GridSpec for the 2x5 grid
    gs_right = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=gs[:, 1])

    # Create each of the 10 plots for prediction distributions
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
            index += 1

    prediction_lines = [ax.plot([], [], lw=2)[0] for ax in prediction_axes]

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    
    # ANIMATION FUNCTIONS
    def init():
        # Tracking data init
        scat_home.set_offsets(np.array([]).reshape(0, 2))
        scat_away.set_offsets(np.array([]).reshape(0, 2))
        scat_ball.set_offsets(np.array([]).reshape(0, 2))

        # Predictions init
        for line in prediction_lines:
            line.set_data([0], [0])

    # Update function
    def update(frame):
        
        # Update video
        ret, video_frame = capture.read()
        if not ret:
            print("Failed to grab frame.")
            return
        video_ax.imshow(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))

        adjusted_frame = frame // update_interval_prediction
        beginning = int(np.max([adjusted_frame+1-1500, 0]))

        # Update tracking data
        scat_home.set_offsets(coords[adjusted_frame,:,:11].T)
        scat_away.set_offsets(coords[adjusted_frame,:,11:].T)
        scat_ball.set_offsets(ball_coords[adjusted_frame])
         
        # Update predictions
        for i, (line, ax) in enumerate(zip(prediction_lines, prediction_axes)):
            line.set_data(x_time[beginning : adjusted_frame+1], visualiser.segmentation[beginning:adjusted_frame+1, i])
            ax.set_xlim(x_time[beginning], x_time[adjusted_frame+1])

    # use animation 
    ani = animation.FuncAnimation(fig=fig, func=update, frames=sys_args.threshold, init_func=init, interval=1000/fps)

    writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-preset', 'medium', '-crf', '23'])
    ani.save(f"{sys_args.output}/GameVisualisationPrediction.mp4", writer=writer)
    capture.release()

if __name__ == '__main__':
    main()
