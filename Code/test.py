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

from matplotlib.collections import LineCollection
pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)

# get scalars to represent players position on the map
scalars = (pitch.dim.pitch_length, pitch.dim.pitch_width)
coords = dataset.matrix.copy()
coords[:,0,:] = coords[:,0,:]*scalars[0]
coords[:,1,:] = coords[:,1,:]*scalars[1]
ball_coords = dataset.ball_coords.copy()
ball_coords[:,0] = ball_coords[:,0]*scalars[0]
ball_coords[:,1] = ball_coords[:,1]*scalars[1]

# create base animation
fig, ax = plt.subplots()
pitch.draw(ax=ax)

# create an empty collection for edges
edge_collection = LineCollection([], colors='white', linewidths=0.5)
# add the collection to the axis
ax.add_collection(edge_collection)

# base scatter boxes
scat_home = ax.scatter([], [], c="r", s=50)
scat_away = ax.scatter([], [], c="b", s=50)
scat_ball = ax.scatter([], [], c="black", s=50)
# base title
timestamp = ax.set_title(f"Timestamp: {0}")
# vectors
# include vectors in the animation
movement_angles_home = dataset.matrix[0, 6,:11]  
movement_angles_away = dataset.matrix[0, 6,11:] 

movement_vectors_home = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_home])
movement_vectors_away = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_away])

quiver_home = ax.quiver(coords[0,0,:11], coords[0,1,:11], movement_vectors_home[:,0], movement_vectors_home[:,1], width=0.002)
quiver_away = ax.quiver(coords[0,0,11:], coords[0,1,11:], movement_vectors_away[:,0], movement_vectors_away[:,1], width=0.002)

# def init():
#     scat_home.set_offsets(np.array([]).reshape(0, 2))
#     scat_away.set_offsets(np.array([]).reshape(0, 2))
#     scat_ball.set_offsets(np.array([]).reshape(0, 2))
#     quiver_home.set_offsets(np.array([]).reshape(0, 2))
#     quiver_away.set_offsets(np.array([]).reshape(0, 2))
#     return (scat_home,scat_away,scat_ball)
# get update function
def update(frame):
    scat_home.set_offsets(coords[frame,0:2,:11].T)
    scat_away.set_offsets(coords[frame,0:2,11:].T)
    scat_ball.set_offsets(ball_coords[frame])
    # convert seconds to minutes and seconds
    minutes, seconds = divmod(dataset.game_details[frame], 60)
    # format the output as mm:ss
    formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
    timestamp.set_text(f"Timestamp: {formatted_time}")
    
    # include vectors in the animation
    movement_angles_home = dataset.matrix[frame, 6,:11]  
    movement_angles_away = dataset.matrix[frame, 6,11:] 

    movement_vectors_home = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_home])
    movement_vectors_away = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_away])
 
    quiver_home.set_UVC(movement_vectors_home[:,0],movement_vectors_home[:,1])
    quiver_home.set_offsets(coords[frame,0:2,:11].T)
    quiver_away.set_UVC(movement_vectors_away[:,0],movement_vectors_away[:,1])
    quiver_away.set_offsets(coords[frame,0:2,11:].T)
    # quiver_home = ax.quiver(coords[frame,0,:11], coords[frame,1,:11], movement_vectors_home[:,0], movement_vectors_home[:,1])
    # quiver_away = ax.quiver(coords[frame,0,11:], coords[frame,1,11:], movement_vectors_away[:,0], movement_vectors_away[:,1])

    return (scat_home, scat_away, scat_ball, timestamp)
# get number of iterations
iterartions = dataset.matrix.shape[0]

# set order of the plot components
scat_home.set_zorder(3) 
scat_away.set_zorder(3)
scat_ball.set_zorder(3)

# use animation 
ani = animation.FuncAnimation(fig=fig, func=update, frames=iterartions, interval=100)

plt.show()
plt.close()

frame = 100

pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
fig, ax = plt.subplots()
pitch.draw(ax=ax)

# create an empty collection for edges
edge_collection = LineCollection([], colors='white', linewidths=0.5)
# add the collection to the axis
ax.add_collection(edge_collection)

# base scatter boxes
scat_home = ax.scatter([], [], c="r", s=50)
scat_away = ax.scatter([], [], c="b", s=50)
scat_ball = ax.scatter([], [], c="black", s=50)
# base title
timestamp = ax.set_title(f"Timestamp: {0}")
# vectors
quiver_home = ax.quiver([], [], [], [])
quiver_away = ax.quiver([], [], [], [])


scat_home.set_offsets(coords[frame,0:2,:11].T)
scat_away.set_offsets(coords[frame,0:2,11:].T)
scat_ball.set_offsets(ball_coords[frame])
# convert seconds to minutes and seconds
minutes, seconds = divmod(dataset.game_details[frame], 60)
# format the output as mm:ss
formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
timestamp.set_text(f"Timestamp: {formatted_time}")

# include vectors in the animation
movement_angles_home = dataset.matrix[frame, 6, :11]  
movement_angles_away = dataset.matrix[frame, 6, 11:] 

movement_vectors_home = np.array([[np.cos(angle), np.sin(angle)] for angle in movement_angles_home])
movement_vectors_away = np.array([[np.cos(angle), np.sin(angle)] for angle in movement_angles_away])
movement_vectors_home[:,0]
positions_home = coords[frame, 0:2, :11].T
positions_away = coords[frame, 0:2, 11:].T

# Update quiver (movement direction) for players
quiver_home.set_UVC(movement_vectors_home[:,0], movement_vectors_home[:,1])
quiver_home.set_offsets(coords[frame,0:2,:11])  # Adjust if necessary
quiver_away.set_UVC(movement_vectors_away)
quiver_away.set_offsets(coords[frame,0:2,11:].T)

pitch = Pitch(pitch_color='grass', line_color='white', stripe=True)
fig, ax = plt.subplots()
pitch.draw(ax=ax)
ax.quiver(positions_home[:,0], positions_home[:,1], movement_vectors_home[:, 0], movement_vectors_home[:, 1])

plt.show()
len(quiver_home.get_p())
coords.shape

































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
