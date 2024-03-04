import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from kloppy import TRACABSerializer, to_pandas
from kloppy.domain.models.common import Point, Player, Team
from kloppy.domain.models.tracking import PlayerData, TrackingDataset, Frame
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mplsoccer.pitch import Pitch
import logging
import seaborn as sns
from matplotlib.collections import LineCollection
from helpers.classes import EVENT_DICTIONARY_V2_ALIVE as classes_enc

class DatasetPreprocessor:
    """ Data class used for preprocessing of the positional data and generation of the annotations"""
    def __init__(self, sample_rate: float, name, alive: bool = False):
        
        self.alive = alive
        self.dataset = None
        self.sample_rate = sample_rate
        self.switch_frames = None
        self.first_period_cntr = 0
        self.second_period_cntr = 0

        # parse helpers
        self.pitch = None
        self.fps = 4   

        # helpers for data preprocessing
        if name[:3] == "BEL":
            self.belgium_role = "home"
        else:
            self.belgium_role = "away"

        # missing players discovered during data exploration
        self.name = name
        self.red_card_games = ['BEL-GRE','BEL-RUS', 'AUS-BEL']
        self.missing_players_games = ['NED-BEL', 'ICE-BEL']


    def open_dataset(self, datafilepath: str, metafilepath: str, annotatedfilepath: str) -> TrackingDataset:
        """Parse file using kloppy lib and create unique player df

        Args:
            datafilepath (str): filepath
            metafilepath (str): filepath
            
        Returns:
            TrackingDataset: kloppy dataset
        """
        serializer = TRACABSerializer()
        with open(datafilepath, "rb") as data, open(metafilepath, "rb") as meta:
            dataset = serializer.deserialize(
                inputs={
                    "raw_data": data,
                    "metadata": meta,
                },
                options={
                    "sample_rate": self.sample_rate,
                    "only_alive": self.alive,
                }
            )

        self.dataset = dataset
        self.pitch = dataset.metadata.pitch_dimensions
        self.fps = int(self.sample_rate * dataset.metadata.frame_rate)
        
        self.first_half_ann = np.load(annotatedfilepath)["array1"]
        self.second_half_ann = np.load(annotatedfilepath)["array2"]
        
        # get belgium coords to determine their field side
        belgium_x_coord = [playerdata.coordinates.x for player, playerdata in dataset.frames[0].players_data.items() if player.player_id[0:4]==self.belgium_role]
        self.belgium_field_part = "left" if min(belgium_x_coord)<0.4 else "right"
    
    def substitution_detection(self) -> dict:
        """Find the frames per team where there was a substitution

        Returns:
            dict[str, dict]: 
                keys: home, away; 
                values: dict[int, list]: 
                    keys: frames when substitution occured
                    values: list of tuples consisting pairs (swapped, swapping)
        """
        dataset = self.dataset
        frames = {
            "home": {},
            "away": {}
        }
        old_players = list(map(lambda x: x.player_id, dataset.frames[0].players_coordinates.keys()))

        for index, frame in enumerate(dataset.frames):
            if self.alive:
                if frame.ball_state.value != "alive":
                    continue
            current_players = list(map(lambda x: x.player_id, frame.players_coordinates.keys()))
            # detect player switches
            for team_id in ["home", "away"]:
                current_players_team = [p for p in current_players if team_id in p]
                old_players_team = [p for p in old_players if team_id in p]
                swapped_players = list(set(old_players_team) - set(current_players_team))
                swapping_players = list(set(current_players_team) - set(old_players_team))

                if len(swapped_players) > 0:
                    frames[team_id][index] = list(zip(swapped_players, swapping_players))
            old_players = current_players  
        self.switch_frames = frames
        return frames

    
    def generate_encodings(self):
        self.player_encoder = {}
        self.player_decoder = {}
        favourable_counter = 0
        opposing_counter = 11
        for frame in self.dataset.frames:
            for player in frame.players_data:
                if player.player_id not in self.player_encoder.keys():
                    # check if belgium player to assign first eleven encoders
                    if player.player_id[0:4] == self.belgium_role:
                        self.player_encoder[player.player_id] = favourable_counter
                        self.player_decoder[favourable_counter] = player.player_id
                        favourable_counter += 1
                    else:
                        self.player_encoder[player.player_id] = opposing_counter
                        self.player_decoder[opposing_counter] = player.player_id
                        opposing_counter += 1
            if len(self.player_encoder.keys())==22:
                break
    
    def _get_player_presence(self):
        # get starting players and note their occurence
        starting_players = list(map(lambda x: x.player_id, self.dataset.frames[0].players_coordinates.keys()))
        starting_frame = [[0] for x in range(len(starting_players))]
        frame_borders = dict(zip(starting_players, starting_frame))
        # generate substitutions
        substitutions = self.substitution_detection()
        # record frames when swap occured
        for vals in substitutions.values():
            for frame_index, frame_swaps in vals.items():
                for swap in frame_swaps:
                    frame_borders[swap[0]].append(frame_index)
                    frame_borders[swap[1]] = [frame_index]
        # assign last fram for players present at the end
        for _, val in frame_borders.items():
            if len(val) == 1:
                val.append(len(self.dataset.frames)-1)
        return frame_borders


    def _get_player_coordinates(self, frame):  
        coord_matrix = np.zeros((22,2))
        ball_coord = np.array([frame.ball_coordinates.x, frame.ball_coordinates.y])
        ball_dist = np.zeros((22,1))
        player_speed = np.zeros((22,1))
        for player, playerdata in frame.players_data.items():
            if (self.belgium_field_part == "left") & (frame.period.id == 1):
                player_coord = [playerdata.coordinates.x, playerdata.coordinates.y]
                ball_coord = np.array([frame.ball_coordinates.x, frame.ball_coordinates.y])
                
            elif (self.belgium_field_part == "left") & (frame.period.id == 2):
                player_coord = [1-playerdata.coordinates.x, 1-playerdata.coordinates.y]
                ball_coord = np.array([1-frame.ball_coordinates.x, 1-frame.ball_coordinates.y])
                
            elif (self.belgium_field_part == "right") & (frame.period.id == 1):
                player_coord = [1-playerdata.coordinates.x, 1-playerdata.coordinates.y]
                ball_coord = np.array([1-frame.ball_coordinates.x, 1-frame.ball_coordinates.y])
                
            else:
                player_coord = [playerdata.coordinates.x, playerdata.coordinates.y]
                ball_coord = np.array([frame.ball_coordinates.x, frame.ball_coordinates.y])
            
            coord_matrix[self.player_encoder[player.player_id]] = player_coord

            if ball_coord[0] > player_coord[0]:
                ball_dist[self.player_encoder[player.player_id]] = np.linalg.norm(player_coord - ball_coord)
            else:
                ball_dist[self.player_encoder[player.player_id]] = -np.linalg.norm(player_coord - ball_coord)

            player_speed[self.player_encoder[player.player_id]] = playerdata.speed
        return np.hstack((coord_matrix,ball_dist,player_speed))
    
    def _get_ball_coordinates(self, frame):
        if (self.belgium_field_part == "left") & (frame.period.id == 1):
            ball_coord = [frame.ball_coordinates.x, frame.ball_coordinates.y]
            
        elif (self.belgium_field_part == "left") & (frame.period.id == 2):
            ball_coord = [1-frame.ball_coordinates.x, 1-frame.ball_coordinates.y]
            
        elif (self.belgium_field_part == "right") & (frame.period.id == 1):
            ball_coord = [1-frame.ball_coordinates.x, 1-frame.ball_coordinates.y]
            
        else:
            ball_coord = [frame.ball_coordinates.x, frame.ball_coordinates.y]
        return ball_coord
    
    def _get_game_details(self,frame):
        if frame.period.id == 1:
            self.first_period_cntr += 1
            return frame.timestamp
        elif frame.period.id == 2:
            self.second_period_cntr += 1
            return frame.timestamp+2700
        
    def _player_violation(self):
        player_violation = []
        for frame in self.dataset.frames:
            if frame.period.id == 1:
                timestamp = frame.timestamp
            elif frame.period.id == 2:
                timestamp = frame.timestamp+2700
            # check if all players are present in the current frame:
            if len(frame.players_coordinates.keys())<22:
                player_violation.append(timestamp)
        return player_violation
    
    def _generate_node_features(self):
        ''' Generates the nodes' features for each frame. The final result is obtained in the self.matrix.
            It is the 3d matrix which can be desribed as follows:
            - 1st axis: Frames
            - 2nd axis: Features
            - 3rd axis: Players
            The order of features is as follows: 
            x-position, y-position, distance to the ball, speed, x-direction, y-direction, movement direction (in radians), order of average position, team affiliation, red card flag.
            There is also order of players in the matrix. First eleven are the Belgian players, the last 11 are the opponents players.
        '''
        try:
            self.player_encoder
        except AttributeError:
            self.generate_encodings()

        self.matrix = []
        self.ball_coords = []
        self.game_details = []
        player_violation = []

        # find substitutions
        self.substitution_detection()
        for index, frame in enumerate(self.dataset.frames):
            # check if all players are present in the current frame:
            if len(frame.players_coordinates.keys())<22:
                player_violation.append(len(frame.players_coordinates.keys()))
            
            # focus only on the alive phases
            if self.alive:
                if frame.ball_state.value != "alive":
                    continue
            
            # check if players should be swapped
            if index in self.switch_frames["home"].keys():
                swap_list = self.switch_frames["home"][index]
                for swap in swap_list:
                    self.player_decoder[self.player_encoder[swap[0]]] = swap[1]
                    self.player_encoder[swap[1]] = self.player_encoder[swap[0]]
            if index in self.switch_frames["away"]:
                swap_list = self.switch_frames["away"][index]
                for swap in swap_list:
                    self.player_decoder[self.player_encoder[swap[0]]] = swap[1]
                    self.player_encoder[swap[1]] = self.player_encoder[swap[0]]
            
            # generates coordinates
            frame_features = self._get_player_coordinates(frame)
            self.matrix.append(frame_features)
            # get ball position
            self.ball_coords.append(self._get_ball_coordinates(frame))
            # get timestamp 
            self.game_details.append(self._get_game_details(frame))

        self.matrix = np.array(self.matrix)
        self.ball_coords = np.array(self.ball_coords)
        self.game_details = np.array(self.game_details)

        # get direction vector for x-coords
        self.matrix = np.transpose(self.matrix, axes=[0, 2, 1])
        x_coords = self.matrix[:,0,:]
        x_directions = np.diff(x_coords, axis=0)
        x_directions = np.expand_dims(np.vstack((np.zeros((1,self.matrix.shape[2])),x_directions)), axis=1)
        
        # get direction vector for y-coords
        y_coords = self.matrix[:,1,:]
        y_directions = np.diff(y_coords, axis=0)
        y_directions = np.expand_dims(np.vstack((np.zeros((1,self.matrix.shape[2])),y_directions)), axis=1)
        
        # get movement angle in radians
        movement_direction = np.arctan2(y_directions, x_directions)
        
        # get avg positions accumulated
        accumulated_xcoordinates = np.cumsum(x_coords, axis=0)
        nr_of_frames = np.arange(1,self.matrix.shape[0]+1,1)
        average_xcoordinates = (accumulated_xcoordinates.T / nr_of_frames).T
        avg_position_fav = np.argsort(average_xcoordinates[:,0:11], axis=1).argsort(axis=1)
        avg_position_opsing = np.argsort(average_xcoordinates[:,11:], axis=1).argsort(axis=1)
        avg_position_total = np.expand_dims(np.hstack((avg_position_fav,avg_position_opsing)), axis=1)
        
        # get team affiliation
        team_affiliation = np.concatenate((np.zeros((self.matrix.shape[0],11)),np.ones((self.matrix.shape[0],11))), axis=1)
        team_affiliation = np.expand_dims(team_affiliation, axis=1)
        
        # add red card flag
        red_flag = np.zeros((self.matrix.shape[0], 1, self.matrix.shape[2]))
        if self.name in self.red_card_games:
            red_card_players = []
            red_card_frames = []
            for team_vals in self.switch_frames.values():
                for key, vals in team_vals.items():
                    if len(vals) == 0:
                        current_frame_players = set([player.player_id for _, player in enumerate(self.dataset.frames[key].players_coordinates)])
                        prev_frame_players = set([player.player_id for _, player in enumerate(self.dataset.frames[key-1].players_coordinates)])
                        red_card_player = prev_frame_players - current_frame_players
                        red_card_players.append(red_card_player.pop())
                        red_card_frames.append(key)
            for red_frame, red_player in zip(red_card_frames, red_card_players):
                red_flag[red_frame:, :, self.player_encoder[red_player]] = 1
        # concatenate new variables
        self.matrix = np.concatenate((self.matrix, x_directions, y_directions, movement_direction, avg_position_total, team_affiliation, red_flag), axis=1)
        
        # mising player cases discovered at data preprocessing part
        if self.name in self.missing_players_games:
            player_violation_frame_indx = []
            for index, frame in enumerate(self.dataset.frames):
                # check if all players are present in the current frame:
                if len(frame.players_coordinates.keys())<22:
                    player_violation_frame_indx.append(index)
            self.matrix = np.delete(self.matrix, player_violation_frame_indx, axis=0)

        return player_violation

    def _generate_edges(self, threshold:float=0.2):
        """
        Generates adjacency matrix
        """
        try:
            self.matrix
        except AttributeError:
            _ = self._generate_node_features()

        frame_dim, _, player_dim = self.matrix.shape
        
        # calculate distances at x-axis between each player for each frame 
        x_coords = self.matrix[:,0,:]
        x_coords = x_coords.reshape(frame_dim,1,player_dim)
        x_dist = x_coords  - np.transpose(x_coords,(0,2,1))
        
        # calculate distances at y-axis between each player for each frame 
        y_coords = self.matrix[:,1,:]
        y_coords = y_coords.reshape(frame_dim,1,player_dim)
        y_dist = y_coords - np.transpose(y_coords,(0,2,1))

        # get final distances
        difference = x_dist**2+y_dist**2
        distance = np.sqrt(difference)

        # determine connections
        # distance = np.where(distance < threshold, 1, 0)
        distance = np.where(distance < threshold, distance, 0)
        distance = 1-distance

        # fill diagonal for each frame with ones
        player_indx = np.arange(player_dim)
        distance[:, player_indx, player_indx] = 1
        self.edges = distance
    
    def _synchronize_annotations(self, focused_annotation):
        # synchronize annotations
        first_half_ann = self.first_half_ann[:self.first_period_cntr]
        second_half_ann = self.second_half_ann[:self.second_period_cntr]
        self.annotations = np.concatenate([first_half_ann, second_half_ann])
        
        if focused_annotation is not None:
            selected_annotations = self.annotations[:, classes_enc[focused_annotation]]
            self.annotations = np.expand_dims(selected_annotations, axis=1)
        # genereate annotations for frames that nothing happened
        # none_ann_vector = np.all(self.annotations == 0, axis=1).astype(int)
        # self.annotations = np.concatenate([none_ann_vector.reshape(-1, 1), self.annotations], axis=1)


    def _generate_annotaions(self):
        alive = np.array([int(frame.ball_state.value=="alive") for frame in self.dataset.frames])
        dead = np.ones(alive.shape) - alive
        # self.annotations = np.expand_dims(dead, axis=1)
        self.annotations = np.vstack((alive, dead)).T

    def animate_game(self, edge_threshold:float=None, direction:bool=False, frame_threshold=None, save_dir=None, interval=1, annotation="Shot"):
        try:
            self.matrix
        except AttributeError:
            _ = self._generate_node_features()

        try:
            self.annotations
        except AttributeError:
            self._synchronize_annotations()

        if edge_threshold:
            self._generate_edges(threshold=edge_threshold)

        # Generate pitch
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
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 15))
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
        # base for movement direction
        if direction:
            movement_angles_home = self.matrix[0, 6,:11]  
            movement_angles_away = self.matrix[0, 6,11:] 

            movement_vectors_home = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_home])
            movement_vectors_away = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_away])

            quiver_home = ax.quiver(coords[0,0,:11], coords[0,1,:11], movement_vectors_home[:,0], movement_vectors_home[:,1], width=0.002)
            quiver_away = ax.quiver(coords[0,0,11:], coords[0,1,11:], movement_vectors_away[:,0], movement_vectors_away[:,1], width=0.002)
        
        # PLOT ANNOTATIONS
        ann = ax2.plot(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold),classes_enc[annotation]], label=annotation)
        ax2.set_title(annotation)
        ax2.legend()

        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            ann[0].set_data(np.arange(0, int(frame_threshold)), self.annotations[:int(frame_threshold), classes_enc[annotation]])
            return (scat_home,scat_away,scat_ball)
        
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])
            ann[0].set_data(np.arange(0, int(frame) + 1), self.annotations[:int(frame)+1, classes_enc[annotation]])
            # convert seconds to minutes and seconds
            minutes, seconds = divmod(self.game_details[frame], 60)
            # format the output as mm:ss
            formatted_time = f"{int(np.round(minutes, 0))}:{int(np.round(seconds, 0))}"
            timestamp.set_text(f"Timestamp: {formatted_time}")
            
            # include edges in the animation
            if edge_threshold:
                segments = []
                row_indices, col_indices = np.where(self.edges[frame] == 1)
                for i, j in zip(row_indices, col_indices):
                    segments.append([(coords[frame, 0, i], coords[frame, 1, i]),
                                    (coords[frame, 0, j], coords[frame, 1, j])])
                # set all segments at once for the LineCollection
                edge_collection.set_segments(segments)
            
            # include movement directions in the animation
            if direction:
                movement_angles_home = self.matrix[frame, 6,:11]  
                movement_angles_away = self.matrix[frame, 6,11:] 

                movement_vectors_home = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_home])
                movement_vectors_away = np.array([[np.cos(angle), -np.sin(angle)] for angle in movement_angles_away])
            
                quiver_home.set_UVC(movement_vectors_home[:,0],movement_vectors_home[:,1])
                quiver_home.set_offsets(coords[frame,0:2,:11].T)
                quiver_away.set_UVC(movement_vectors_away[:,0],movement_vectors_away[:,1])
                quiver_away.set_offsets(coords[frame,0:2,11:].T)

            return (scat_home, scat_away, scat_ball, timestamp)
        
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


    def generate_heatmaps(self):
        try:
            self.matrix
        except AttributeError:
            _ = self._generate_node_features()

        pitch = Pitch(pitch_color="grass", line_color='white',
              stripe=False) 
        # get scalars to represent players position on the map
        scalars = (pitch.dim.pitch_length, pitch.dim.pitch_width)
        # get dictionary discribing during which frame the player occured
        frame_borders = self._get_player_presence()

        # Determine Belgium and opponent players
        if self.belgium_role == "home":
            belgium_players = [home for home in frame_borders.keys() if "home" in home]
            belgium_cnt = len(belgium_players)
            opponent_players = [away for away in frame_borders.keys() if "away" in away]
            opponent_cnt = len(opponent_players)
        else:
            belgium_players = [away for away in frame_borders.keys() if "away" in away]
            belgium_cnt = len(belgium_players)
            opponent_players = [home for home in frame_borders.keys() if "home" in home]
            opponent_cnt = len(opponent_players)
        # plot heatmaps for Belgium 
        nrows = int(np.ceil(belgium_cnt/4))

        fig, axs = pitch.draw(nrows=nrows, ncols=4, figsize=(8, 6))
        player_cntr = 0
        for row in range(nrows):
            for col in range(4):
                player = belgium_players[player_cntr]
                player_cntr += 1
                player_frame_borders = frame_borders[player]
                player_id = self.player_encoder[player]
                x = self.matrix[player_frame_borders[0]:player_frame_borders[1],0,player_id]*scalars[0]
                y = self.matrix[player_frame_borders[0]:player_frame_borders[1],1,player_id]*scalars[1]

                sns.kdeplot(x=x, y=y, fill=True, cmap="coolwarm", n_levels=50, ax=axs[row,col], zorder=0)
                axs[row,col].set_aspect('equal')
                axs[row,col].set_title(player)
        plt.show()



# # Your matrix
# matrix = np.array([[0, 0, 0],
#                    [1, 0, 0],
#                    [0, 0, 1]])

# # Specify the row for which you want to create the vector
# row_index = 1

# # Create the vector based on the specified row
# vector = np.all(matrix == 0, axis=1).astype(int)

# np.concatenate([vector.reshape(-1, 1), matrix], axis=1)