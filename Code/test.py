import os
os.chdir("Code")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mplsoccer.pitch import Pitch
from DataPreprocessing import Dataset
from FileFinder import find_files
import logging
from tqdm import tqdm
from DataManager import DataManager

logging.basicConfig(level=logging.INFO)
files = find_files("../football_games")

dataset = Dataset(1/5, files[0].home, alive=True)
dataset.open_dataset(files[0].datafile, files[0].metafile)
dataset._generate_node_features()
dataset._generate_edges()
dataset.animate_game(edge_threshold=0.2, frame_threshold=None, save_dir=None, interval=1)
dataset.edges[0]

dataset.matrix[0,0,0]


include_edges = True
frame_threshold = None
# Generate pitch
pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
fig, ax = pitch.draw()

# get pitch size
x_axis_size = ax.get_xlim()
y_axis_size = ax.get_ylim()

# get scalars to represent players position on the map
scalars = (x_axis_size[0]+x_axis_size[1], y_axis_size[0]+y_axis_size[1])
coords = dataset.matrix.copy()
coords[:,0,:] = coords[:,0,:]*scalars[0]
coords[:,1,:] = coords[:,1,:]*scalars[1]
ball_coords = dataset.ball_coords.copy()
ball_coords[:,0] = ball_coords[:,0]*scalars[0]
ball_coords[:,1] = ball_coords[:,1]*scalars[1]

# create animation
fig, ax = pitch.draw()
from matplotlib.collections import LineCollection
# Create an empty LineCollection for edges
edge_collection = LineCollection([], colors='grey')
# Add the LineCollection to the axis
ax.add_collection(edge_collection)
scat_home = ax.scatter([], [], c="r", s=50)
scat_away = ax.scatter([], [], c="b", s=50)
scat_ball = ax.scatter([], [], c="black", s=50)
timestamp = ax.set_title(f"Timestamp: {0}")

def init():
    scat_home.set_offsets(np.array([]).reshape(0, 2))
    scat_away.set_offsets(np.array([]).reshape(0, 2))
    scat_ball.set_offsets(np.array([]).reshape(0, 2))
    return (scat_home,scat_away,scat_ball)

def update(frame):
    scat_home.set_offsets(coords[frame,:,:11].T)
    scat_away.set_offsets(coords[frame,:,11:].T)
    scat_ball.set_offsets(ball_coords[frame])
    # Convert seconds to minutes and seconds
    minutes, seconds = divmod(dataset.game_details[frame], 60)
    # Format the output as mm:ss
    formatted_time = f"{int(np.round(minutes,0))}:{int(np.round(seconds,0))}"
    timestamp.set_text(f"Timestamp: {formatted_time}")

    if include_edges:
        edge_collection.set_segments([])
        row_indices, col_indices = np.where(dataset.edges[frame] == 1)
        for i, j in zip(row_indices, col_indices):
            segments = np.array([[coords[frame, 0, i], coords[frame, 1, i]],
                                 [coords[frame, 0, j], coords[frame, 1, j]]])
            edge_collection.set_segments([segments])

    return (scat_home,scat_away,scat_ball,timestamp)

iterartions = dataset.matrix.shape[0]

ani = animation.FuncAnimation(fig=fig, func=update, frames=iterartions, init_func=init, interval=10)
plt.show()


# create base animation
fig, ax = plt.subplots()
pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
pitch.draw(ax=ax)


# create an empty collection for edges
edge_collection = LineCollection([], colors='white', linewidths=0.5)
# add the collection to the axis
ax.add_collection(edge_collection)

scat_home = ax.scatter([], [], c="r", s=50)
scat_away = ax.scatter([], [], c="b", s=50)
scat_ball = ax.scatter([], [], c="black", s=50)
timestamp = ax.set_title(f"Timestamp: {0}")

def init():
    scat_home.set_offsets(np.array([]).reshape(0, 2))
    scat_away.set_offsets(np.array([]).reshape(0, 2))
    scat_ball.set_offsets(np.array([]).reshape(0, 2))
    return (scat_home, scat_away, scat_ball)

def update(frame):
    scat_home.set_offsets(coords[frame,:,:11].T)
    scat_away.set_offsets(coords[frame,:,11:].T)
    scat_ball.set_offsets(ball_coords[frame])
    # Convert seconds to minutes and seconds
    minutes, seconds = divmod(dataset.game_details[frame], 60)
    # Format the output as mm:ss
    formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
    timestamp.set_text(f"Timestamp: {formatted_time}")

    if include_edges:
        segments = []
        row_indices, col_indices = np.where(dataset.edges[frame] == 1)
        for i, j in zip(row_indices, col_indices):
            segments.append([(coords[frame, 0, i], coords[frame, 1, i]),
                             (coords[frame, 0, j], coords[frame, 1, j])])
        # Set all segments at once for the LineCollection
        edge_collection.set_segments(segments)

    return (scat_home, scat_away, scat_ball, timestamp)

pitch.set_zorder(1)  # Set the pitch zorder to 1
scat_home.set_zorder(3)  # Set the scatter plots zorder to 3
scat_away.set_zorder(3)
scat_ball.set_zorder(3)

iterations = dataset.matrix.shape[0]

ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, init_func=init, interval=10)
plt.show()
