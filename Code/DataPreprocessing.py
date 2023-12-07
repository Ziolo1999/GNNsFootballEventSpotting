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

class Dataset:
    """ Data class which will be used to preprocess positional data """
    def __init__(self, sample_rate: float, name, alive: bool = False) -> None:
        self.alive = alive
        self.dataset = None
        self.sample_rate = sample_rate
        self.switch_frames = None
        self.role_cluster = {}
        self.role_amount = 0

        # dataframes
        self.home_df = None # df from view of home team
        self.away_df = None # df from view of away team
        self.unique_player_df = None # df with all players
        self.role_player_df = None # df with cluster nr for each player currently on field
        self.phase_df = None

        # parse helpers
        self.ball_dist = None
        self.ball_owner_cutoff = 1
        self.parsed_result = {}
        self.parse_data = []
        self.pitch = None
        self.skip_counter = 0
        self.fps = 4   

        # helpers for data preprocessing
        if name[:3] == "BEL":
            self.belgium_role = "home"
        else:
            self.belgium_role = "away"
        # discovered during data exploration
        self.name = name
        self.red_card_games = ['BEL-GRE','BEL-RUS', 'AUS-BEL']
        self.missing_players_games = ['NED-BEL', 'ICE-BEL']


    def open_dataset(self, datafilepath: str, metafilepath: str) -> TrackingDataset:
        """Parse file using kloppy lib and create unique player df

        Args:
            datafilepath (str): filepath
            metafilepath (str): filepath
            alive (bool, optional): Use only frames were ball is alive. Defaults to False.

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
        self.unique_player_df = to_pandas(dataset)
        self.unique_player_df = self.unique_player_df.loc[self.unique_player_df["ball_state"] == "alive"] 
        self.pitch = dataset.metadata.pitch_dimensions
        self.fps = int(self.sample_rate * dataset.metadata.frame_rate)
        # get belgium coords to determine their field side
        belgium_x_coord = [playerdata.coordinates.x for player, playerdata in dataset.frames[0].players_data.items()  if player.player_id[0:4]==self.belgium_role]
        self.belgium_field_part = "left" if min(belgium_x_coord)<0.4 else "right"
    
    def substitution_detection(self) -> dict:
        """Find the frames per team where there was a substitution

        Returns:
            dict[str, list]: keys: home, away; values: list of frames of substitutions
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
        # print(f"All players detected at the {frame.timestamp} timestamp")
    
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
                player_coord = [1-playerdata.coordinates.x, playerdata.coordinates.y]
                ball_coord = np.array([1-frame.ball_coordinates.x, frame.ball_coordinates.y])
                
            elif (self.belgium_field_part == "right") & (frame.period.id == 1):
                player_coord = [1-playerdata.coordinates.x, playerdata.coordinates.y]
                ball_coord = np.array([1-frame.ball_coordinates.x, frame.ball_coordinates.y])
                
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
                ball_coord = [1-frame.ball_coordinates.x, frame.ball_coordinates.y]
                
            elif (self.belgium_field_part == "right") & (frame.period.id == 1):
                ball_coord = [1-frame.ball_coordinates.x, frame.ball_coordinates.y]
                
            else:
                ball_coord = [frame.ball_coordinates.x, frame.ball_coordinates.y]
            return ball_coord
    
    def _get_game_details(self,frame):
        if frame.period.id == 1:
            return frame.timestamp
        elif frame.period.id == 2:
            return frame.timestamp + 45
        
    def _player_violation(self):
        player_violation = []
        for frame in self.dataset.frames:
            if frame.period.id == 1:
                timestamp = frame.timestamp
            elif frame.period.id == 2:
                timestamp = frame.timestamp+45
            # check if all players are present in the current frame:
            if len(frame.players_coordinates.keys())<22:
                player_violation.append(timestamp)
        return player_violation
    
    def _generate_node_features(self):
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
        return player_violation

    def _generate_edges(self, threshold:float=0.2):
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
        distance = np.where(distance < threshold, 1, 0)

        # fill diagonal for each frame with ones
        player_indx = np.arange(player_dim)
        distance[:, player_indx, player_indx] = 1
        self.edges = distance

    def animate_game(self, edge_threshold:float=None, frame_threshold=None, save_dir=None, interval=1):
        try:
            self.matrix
        except AttributeError:
            _ = self._generate_node_features()

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
        def init():
            scat_home.set_offsets(np.array([]).reshape(0, 2))
            scat_away.set_offsets(np.array([]).reshape(0, 2))
            scat_ball.set_offsets(np.array([]).reshape(0, 2))
            return (scat_home,scat_away,scat_ball)
        # get update function
        def update(frame):
            scat_home.set_offsets(coords[frame,:,:11].T)
            scat_away.set_offsets(coords[frame,:,11:].T)
            scat_ball.set_offsets(ball_coords[frame])
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
