# %% 
# Reading files
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from kloppy import TRACABSerializer, to_pandas
from kloppy.domain.models.common import Point, Player, Team
from kloppy.domain.models.tracking import PlayerData, TrackingDataset, Frame


# Logging
import logging

# Sklearn
from sklearn.cluster import KMeans

# Plotting / Animating
from mplsoccer.pitch import Pitch
from matplotlib import animation

class PlayerCountException(Exception):
    pass

class TracabDataset:
    """ Data class for parsing an individual match from a tracab file """
    def __init__(self, sample_rate: float) -> None:
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
        self.fps = 4            # default, gets overwritten in open_dataset

    def get_home_df(self):
        if self.home_df is None:
            home = self._parse_tracab()
            self.home_df = pd.DataFrame.from_dict(home)
        return self.home_df

    def get_away_df(self):
        if self.away_df is None:
            away = self._parse_tracab(friend="away", enemy="home")
            self.away_df = pd.DataFrame.from_dict(away)
        return self.away_df

    def open_dataset(self, datafilepath: str, metafilepath: str, alive: bool = False) -> TrackingDataset:
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
                    "only_alive": alive,
                    # "count_dead": True,
                }
            )
            # self.datasets["datafilepath"] = dataset
        self.dataset = dataset
        self.unique_player_df = to_pandas(dataset)
        self.unique_player_df = self.unique_player_df.loc[self.unique_player_df["ball_state"] == "alive"] 
        self.pitch = dataset.metadata.pitch_dimensions
        self.fps = int(self.sample_rate * dataset.metadata.frame_rate)
        return dataset
        
    def find_switch_frames(self) -> dict:
        """Find the frames per team where there was a substitution

        Returns:
            dict[str, list]: keys: home, away; values: list of frames of substitutions
        """
        dataset = self.dataset
        frames = {
            "home": [0],
            "away": [0]
        }
        old_players = list(map(lambda x: x.player_id, dataset.frames[0].players_coordinates.keys()))

        for index, frame in enumerate(dataset.frames):
            if frame.ball_state.value != "alive":
                continue
            current_players = list(map(lambda x: x.player_id, frame.players_coordinates.keys()))

            # detect player switches
            for team_id in ["home", "away"]:
                current_players_team = [p for p in current_players if team_id in p]
                old_players_team = [p for p in old_players if team_id in p]
                missing_players = list(set(old_players_team) - set(current_players_team))

                if len(missing_players) > 0:
                    frames[team_id].append(index)
                
            old_players = current_players  


        frames["away"].append(len(dataset.frames))
        frames["home"].append(len(dataset.frames))
        self.switch_frames = frames
        return frames


    def kmeans_players(self, parsed_result: dict, player_data: list, k=4, prefix=""):
        """Cluster player coordinates into k clusters, cluster nr are ordered on cluster center
        ordered in ascending order (first: lowest x; second: lowest y)

        Args:
            parsed_result (dict): output dictionary
            player_data (list[coordinates]): list of [x,y] coordinates
            k (int): cluster amount
            prefix (str): prefix for output dict keys

        Returns:
            tuple[dict, KMeans]: output dict with cluster info, cluster nr and cluster dist per player, KMeans clustering

        """
        kmeans_player = KMeans(k, tol=1e-4).fit(player_data)
        centers = list(enumerate(kmeans_player.cluster_centers_))
        index_centers_sorted = sorted(centers, key=lambda x: x[1][0])
        index_sorted, centers_sorted = zip(*index_centers_sorted)

        for i, center in enumerate(centers_sorted):
            parsed_result[f"{prefix}cluster{i}_px"] = center[0]
            if len(center) > 1:
                parsed_result[f"{prefix}cluster{i}_py"] = center[1]

        distance_min = np.min(kmeans_player.transform(player_data), axis=1)
        labels = kmeans_player.labels_
        for i, d in enumerate(distance_min):
            parsed_result[f"{prefix}player{i+1}_dist_cluster"] = d
            parsed_result[f"{prefix}player{i+1}_nr_cluster"] = index_sorted.index(labels[i])

        return parsed_result, kmeans_player

    def get_player_roles_df(self) -> pd.DataFrame:
        """ Find approximate position / role of the players => defender, keeper, forward, midfielder, flank
        per PWS (period without switches).
        Useful for when two players switch at once, we need to make sure that player x still fills the same role
        Useful for creating centroids of player roles.
        

        Raises:
            PlayerCountException: occurs when player count != 11

        Returns:
            pd.DataFrame: cluster numbers for each player (col) for each frame (row)
        """
        unique_player_df = self.unique_player_df
        dataset = self.dataset
        if self.switch_frames is None:
            self.find_switch_frames()
        switch_frames = self.switch_frames

        unique_player_df = unique_player_df.reindex(sorted(unique_player_df.columns), axis=1)
        unique_player_df["left"] = unique_player_df["period_id"].apply(lambda x: dataset.metadata.periods[x-1].attacking_direction.value.split('-')[0])
        extra_cols = ["left"]

        role_player_df = unique_player_df.loc[:, ["left"]]

        # helper function to get the currect view on the coordinates
        def convert_coor(row, team):
            if team == row["left"]:
                return row

            for col in row.index:
                if col != "left":
                    row[col] = 1 - row[col]
            return row

        result = {}
        total = (len(switch_frames["home"]) + len(switch_frames["away"]) - 2) * 2
        # with tqdm(total=total, leave=True, desc="Finding player roles") as pbar:

        for coor in ["_x", "_y"]: # compute centers for both x, y coordinates
            unique_player_df_cols = [col for col in unique_player_df.columns if coor in col and not "ball" in col]
            for team in ["home", "away"]: # compute centers seperately for each team
                team_cols = [col for col in unique_player_df_cols if team in col]
                for i in range(len(switch_frames[team]) - 1): # Cluster x, y coor per team, for each sequence without player switches
                    # start end and frame for current sequence
                    start = switch_frames[team][i]
                    end = switch_frames[team][i+1]

                    # slice start, end, convert coordinates, drop missing values
                    df = unique_player_df.loc[start:end-1, team_cols + extra_cols].apply(lambda x: convert_coor(x, team), axis=1).dropna(axis=1)
                    # compute mean x, y coordinate per player
                    df_mean = df.drop(columns=["left"]).mean(axis=0).to_frame(0).T

                    # k = 4 for x, k = 5 for y
                    k = 4 if coor == "_x" else 5
                    # compute cluster centers of the mean coordinate
                    _, c1 = self.kmeans_players(result, df_mean.values[0].reshape(-1, 1), k, prefix=f"{team}{coor}-{i}-")
                    self.role_cluster[team + coor] = c1
                    if len(df_mean.columns) != 11:
                        raise PlayerCountException(f"11 != {len(df_mean.columns)}")
                        
                    for index, player in enumerate(df_mean.columns):
                        role_player_df.loc[start:end-1, f"info_cluster_{player}"] = result[f"{team}{coor}-{i}-player{index+1}_nr_cluster"]
                        # pbar.update(1)

        self.role_player_df = role_player_df
        self.role_amount = int(max(self.role_player_df.drop(columns=["left"]).max()) + 1)
        return role_player_df

    # team role helpers
    def get_current_team_positions(self, frame: int, friend="home"):
        """ Mapping from tracab player name (e.g. away22) to player nr (e.g. player1_friend) and their x and y cluster

        Args:
            frame (int): frame
            friend (str, optional): Which team is considered friendly (home or away). Defaults to "home".

        Returns:
            tuple[dict, dict]: player mapping for both home and away teams
        """
        if self.role_player_df is None:
            self.get_player_roles_df()
        role_player_df = self.role_player_df
        current_player_roles = role_player_df.drop(columns=["left"]).loc[frame].dropna().to_dict()
        unique_players = np.unique(list(map(lambda x: "_".join(x.split('_')[2:4]), current_player_roles.keys())))
        result = {
            "home": [],
            "away": []
        }
        for player in unique_players:
            team = player.split('_')[0]
            result[team].append((player, current_player_roles["info_cluster_" + player + "_x"], current_player_roles["info_cluster_" + player + "_y"]))

        order_home = sorted(result["home"], key= lambda x: (x[1], x[2]))
        order_away = sorted(result["away"], key= lambda x: (x[1], x[2]))

        assert len(order_home) == 11, f"too many/few home players @frame {frame}"
        assert len(order_away) == 11, f"too many/few away players @frame {frame}"
        team_home = {}
        team_away = {}
        for i in range(11):
            extra_home = "friend" if friend == "home" else "enemy"
            extra_away = "enemy" if friend == "home" else "friend"
            team_home[order_home[i][0]] = (f"player{i+1}_{extra_home}", order_home[i][1], order_home[i][2])
            team_away[order_away[i][0]] = (f"player{i+1}_{extra_away}", order_away[i][1], order_away[i][2])

        return team_home, team_away

    def get_roles_in_team(self, team: dict, role: int):
        return [(elem[0], elem[1][0]) for elem in team.items() if elem[1][1] == role]

    def _get_role_count_in_team(self, team: dict, role: int):
        return len([elem for elem in team.values() if elem[1] == role])

    def get_roles_count_in_team(self, team: dict):
        return [self._get_role_count_in_team(team, i) for i in range(self.role_amount)]

    # Player role helpers
    def get_player_roles(self, playerid: str):
        return self.role_player_df[[f"info_cluster_{playerid}_x", f"info_cluster_{playerid}_y"]].value_counts().index.tolist()

    def get_current_player_role(self, playerid: str, frame: int):
        return self.role_player_df.iloc[frame][[f"info_cluster_{playerid}_x", f"info_cluster_{playerid}_y"]].values.tolist()

    # stats
    def get_team_strategy(self, teamid: str):
        test = self.role_player_df.loc[:, ["left"]]
        def count_values(row):
            key, values = np.unique(row.values[~np.isnan(row.values)], return_counts=True)
            return pd.Series(dict(zip(key, values)))
        role_counts = self.role_player_df[[col for col in self.role_player_df.columns if "_x" in col and teamid in col]].apply(count_values, axis=1)
        role_counts["sum"] = role_counts.apply(np.sum, axis=1)
        return role_counts

    def get_team_distances(self, team="friend"):
        cols = [f"player{i+1}_{team}_{c}" for i in range(11) for c in ["px", "py"]] + [f"{team}_centroid_{c}" for c in ["px", "py"]]

            
        df = self.home_df
        d_dict = {}
        # for i in range(11):
            # distances_player = np.linalg.norm(df[])



    # Parsing / Creating df
    ## Ball
    def _ball_metrics(self, frame):
        self.parsed_result["ball_actual_x"] = frame.ball_coordinates.x
        self.parsed_result["ball_actual_y"] = frame.ball_coordinates.y
        self.parsed_result["ball_actual_z"] = frame.ball_coordinates.z
        return self.parsed_result

    def _ball_features(self, frame: Frame, right, friend, period_start):
        """ Retrieve ball statistics from a single frame
        Helper function for _parse_tracab

        Args:
            frame ([type]): [description]
            right ([type]): [description]
            friend ([type]): [description]
            period_start ([type]): [description]

        Returns:
            [type]: [description]
        """
        prev_data = None
        if len(self.parse_data) > 0:
            prev_data = self.parse_data[-1]
        # TODO fix speed with ball z coordinate
        if right == friend:
            self.parsed_result["ball_px"] = 1 - frame.ball_coordinates.x
            self.parsed_result["ball_py"] = 1 - frame.ball_coordinates.y
        else:
            self.parsed_result["ball_px"] = frame.ball_coordinates.x
            self.parsed_result["ball_py"] = frame.ball_coordinates.y

        self.parsed_result["ball_psymy"] = abs(self.parsed_result["ball_py"] - 0.5)
        if prev_data is None or period_start or prev_data["frame_id"] + 10 < frame.frame_id:
            # direction is zero when starting the match, after side switch, throw in / corner kick
            self.parsed_result["ball_dir_x"] = 0
            self.parsed_result["ball_dir_y"] = 0
            self.parsed_result["ball_unnorm_speed"] = 0
            if prev_data is not None and prev_data["frame_id"] + 1 < frame.frame_id:
                self.skip_counter += 1
        else:
            # direction is based on previous position, should be normalized
            diff_x = self.parsed_result["ball_px"] - prev_data["ball_px"]
            diff_y = self.parsed_result["ball_py"] - prev_data["ball_py"]
            diff_z = (self.parsed_result["ball_actual_z"] - prev_data["ball_actual_z"]) / 100 # convert to meters
            som = abs(diff_x) + abs(diff_y) 
            som = 1 if som == 0 else som # prevent divison by zero

            self.parsed_result["ball_dir_x"] = (diff_x / som) / 2
            self.parsed_result["ball_dir_y"] = (diff_y / som) / 2
            self.parsed_result["ball_unnorm_speed"] = np.sqrt((diff_x * self.pitch.length)**2 + (diff_y * self.pitch.width)**2 + (diff_z)**2) * self.fps
            if self.parsed_result["ball_unnorm_speed"] > 140:
                logging.warn(f"ball speed to high")
                logging.info(f"{frame.ball_coordinates} {diff_x}, {diff_y}, {diff_z}, {self.parsed_result['ball_unnorm_speed']}")
                logging.info(f"{self.pitch.width} {self.pitch.length}")
                # print("ball speed to high", frame.ball_coordinates)
                # print(diff_x, diff_y, diff_z, self.parsed_result["ball_unnorm_speed"])

        # self.parsed_result["ball_z"] = frame.ball_coordinates.z
        self.parsed_result["ball_owner"] = frame.ball_owning_team.team_id == friend
        return self.skip_counter

    ## Players

    # TODO total distance between players as a metric
    def _player_stats(self, player: Player, playerdata: PlayerData, teams: dict, period_start, min_team: dict, max_team: dict,
                right: str, friend: str, role_count: dict):
        """ Fill in player statistics values

        Args:
            player (): [description]
            playerdata ([type]): [description]
            teams (dict): [description]
            period_start ([type]): [description]
            min_team (dict): [description]
            max_team (dict): [description]
            right (str): [description]
            friend (str): [description]
            role_count (list): amount of players that have role index

        Returns:
            nothing
        """
        # player
        prev_data = None
        if len(self.parse_data) > 0:
            prev_data = self.parse_data[-1]
        
        col, clusterx, clustery = teams[player.player_id]
        clusterx = int(clusterx)
        point = playerdata.coordinates
        playerteam = player.team.team_id
        friend_enemy = "friend" if playerteam == friend else "enemy"
        self.parsed_result[f"info_{col}"] = player.player_id
        role_count_team = role_count[playerteam]

        if right == friend:
            # invert coordinates so they are always relative to the friend goal, not relative to the field
            # use these to cluster, use actual to animate
            self.parsed_result[col + "_px"] = 1 - point.x
            self.parsed_result[col + "_py"] = 1 - point.y
        else:
            self.parsed_result[col + "_px"] = point.x
            self.parsed_result[col + "_py"] = point.y

        self.parsed_result[col + "_psymy"] = abs(self.parsed_result[col + "_py"] - 0.5)

        self.parsed_result[f"{friend_enemy}_centroid_actual_x"] += point.x / 11
        self.parsed_result[f"{friend_enemy}_centroid_actual_y"] += point.y / 11
        self.parsed_result[f"{friend_enemy}_centroid_px"] += self.parsed_result[col + "_px"] / 11
        self.parsed_result[f"{friend_enemy}_centroid_py"] += self.parsed_result[col + "_py"] / 11
        self.parsed_result[f"{friend_enemy}_centroid_psymy"] += self.parsed_result[col + "_psymy"] / 11
        self.parsed_result[f"{friend_enemy}_centroid_unnorm_speed"] += playerdata.speed / 11
        if clusterx != 0:
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_actual_x"] += point.x / 10
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_actual_y"] += point.y / 10
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_px"] += self.parsed_result[col + "_px"] / 10
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_py"] += self.parsed_result[col + "_py"] / 10
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_psymy"] += self.parsed_result[col + "_psymy"] / 10
            self.parsed_result[f"{friend_enemy}_nokeepercentroid_unnorm_speed"] += playerdata.speed / 10

        min_team[friend_enemy + "x"] = min(min_team[friend_enemy + "x"], self.parsed_result[col + "_px"]) 
        min_team[friend_enemy + "y"] = min(min_team[friend_enemy + "y"], self.parsed_result[col + "_py"]) 
        max_team[friend_enemy + "x"] = max(max_team[friend_enemy + "x"], self.parsed_result[col + "_px"]) 
        max_team[friend_enemy + "y"] = max(max_team[friend_enemy + "y"], self.parsed_result[col + "_py"]) 

        self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_px"] += self.parsed_result[col + "_px"] / role_count_team[clusterx]
        self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_py"] += self.parsed_result[col + "_py"] / role_count_team[clusterx]
        self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_psymy"] += self.parsed_result[col + "_psymy"] / role_count_team[clusterx]
        self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_unnorm_speed"] += playerdata.speed / role_count_team[clusterx]
            # todo add span per role? see above todo 

        ## Metrics -------------------------------------------------
        # Actual player speed, needed for metric
        self.parsed_result[col + "_unnorm_speed"] = playerdata.speed
        # Actual player positions, needed for animations
        self.parsed_result[col + "_actual_x"] = point.x
        self.parsed_result[col + "_actual_y"] = point.y

        # Distance to ball
        diff_x = point.x - self.parsed_result["ball_actual_x"]
        diff_y = point.y - self.parsed_result["ball_actual_y"]
        distance = np.sqrt((diff_x * self.pitch.length)**2 + (diff_y * self.pitch.width)**2)
        if self.ball_dist is None:
            self.ball_dist = (distance, player)
        elif distance < self.ball_dist[0]:
            self.ball_dist = (distance, player)
        
        self.parsed_result[col + "_actual_balldist"] = distance
        self.parsed_result["unnorm_pressure"] += distance
        self.parsed_result["friend_unnorm_pressure"] += distance
        self.parsed_result["enemy_unnorm_pressure"] += distance
        # ------------------------------------------------------------

        ## Features
        if prev_data is None or period_start:
            # direction is zero when starting the match, and after side switch
            self.parsed_result[col + "_dir_x"] = 0
            self.parsed_result[col + "_dir_y"] = 0
        else:
            # direction is based on previous position, should be normalized (we already have speed)
            diff_x = self.parsed_result[col + "_actual_x"] - prev_data[col + "_actual_x"]
            diff_y = self.parsed_result[col + "_actual_y"] - prev_data[col + "_actual_y"]
            som = abs(diff_x) + abs(diff_y)
            som = 1 if som == 0 else som # prevent divison by zero

            self.parsed_result["actual_dir_x"] += (diff_x / som) / 2
            self.parsed_result["actual_dir_y"] += (diff_y / som) / 2

            if right == friend:
                diff_x *= -1
                diff_y *= -1

            self.parsed_result[col + "_dir_x"] = (diff_x / som) / 2
            self.parsed_result[col + "_dir_y"] = (diff_y / som) / 2 

            # direction
            self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_dir_x"] += self.parsed_result[col + "_dir_x"] / role_count_team[clusterx]
            self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_dir_y"] += self.parsed_result[col + "_dir_y"] / role_count_team[clusterx]
            # dist moved between frames
            self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_unnorm_diff"] += np.sqrt(diff_x**2 + diff_y**2) / role_count_team[clusterx]
    
            self.parsed_result[f"{friend_enemy}_actual_dir_x"] += (diff_x / som) / 2
            self.parsed_result[f"{friend_enemy}_actual_dir_y"] += (diff_y / som) / 2



    def _parse_tracab(self, friend="home", enemy="away"):
        """convert tracab dataset into pandas dataframe

        Args:
            dataset (tracab dataset): dataset
            friend (str, optional): id of team that will determine the match's perspective. Defaults to "home".
            enemy (str, optional): id of enemy team. Defaults to "away".
            switch_frames (dict, optional): frames that contain switches / new roles for players. Defaults to {}.

        Returns:
            pd.DataFrame: dataframe of match
        """
        # Todo generalize such that different teams are not necessary
        # Columns with "actual" in their name are filtered out when clustering
        frame_index = 0
        self.parse_data = [] # df in list[dict] form
        old_period = 0
        period_offset = 0
        skip_counter = 0 # how many frame skips occur in the match
        
        ball_dead = False
        match_start = True

        # for frame in tqdm(self.dataset.frames, desc="Transforming dataset", leave=True):
        for frame in self.dataset.frames:
            self.parsed_result = {}
            self.ball_dist = None
            _, right = frame.period.attacking_direction.value.split('-')
            period_start = frame.period != old_period

            if period_start and old_period != 0:
                period_offset = self.parse_data[-1]["info_time"]

            # detect ball state 
            if frame.ball_state.value != "alive":
                ball_dead = True
                frame_index += 1
                continue

            # teams, roles
            if match_start:
                team_home, team_away = self.get_current_team_positions(frame_index)
                teams = {**team_home, **team_away}
                role_count = {
                    "home": self.get_roles_count_in_team(team_home),
                    "away": self.get_roles_count_in_team(team_away),
                }
                match_start = False


            # detect player switches
            for team in ["home", "away"]:
                if frame_index in self.switch_frames[team]:
                    team_home, team_away = self.get_current_team_positions(frame_index)
                    teams = {**team_home, **team_away}
                    role_count = {
                        "home": self.get_roles_count_in_team(team_home),
                        "away": self.get_roles_count_in_team(team_away),
                    }

                assert len(teams) == 22

            self.parsed_result["info_was_dead"] = ball_dead
            if ball_dead:
                ball_dead = False

            # add stats independent of players
            self.parsed_result["frame_id"] = frame.frame_id
            self.parsed_result["info_period"] = frame.period.id
            self.parsed_result["timestamp"] = frame.timestamp
            self.parsed_result["info_time"] = frame.timestamp + period_offset

            self._ball_metrics(frame)
            self._ball_features(frame, right, friend, period_start)

            # Pressure metrics
            self.parsed_result["unnorm_pressure"] = 0
            self.parsed_result["friend_unnorm_pressure"] = 0
            self.parsed_result["enemy_unnorm_pressure"] = 0
            # Direction metrics
            self.parsed_result["actual_dir_x"] = 0
            self.parsed_result["actual_dir_y"] = 0
            self.parsed_result["friend_actual_dir_x"] = 0
            self.parsed_result["friend_actual_dir_y"] = 0
            self.parsed_result["enemy_actual_dir_x"] = 0
            self.parsed_result["enemy_actual_dir_y"] = 0
            # Position, speed metrics (team centroids)
            for friend_enemy in ["friend", "enemy"]:
                self.parsed_result[f"{friend_enemy}_centroid_actual_x"] = 0
                self.parsed_result[f"{friend_enemy}_centroid_actual_y"] = 0
                self.parsed_result[f"{friend_enemy}_centroid_px"] = 0
                self.parsed_result[f"{friend_enemy}_centroid_py"] = 0
                self.parsed_result[f"{friend_enemy}_centroid_psymy"] = 0
                self.parsed_result[f"{friend_enemy}_centroid_unnorm_speed"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_actual_x"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_actual_y"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_px"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_py"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_psymy"] = 0
                self.parsed_result[f"{friend_enemy}_nokeepercentroid_unnorm_speed"] = 0

            # add player stats
            # add x, y, speed, direction per player
            min_team = {"friendx": 2, "friendy": 2, "enemyx": 2, "enemyy": 2}
            max_team = {"friendx": -1, "friendy": -1, "enemyx": -1, "enemyy": -1}
            team_list = ["friend", "enemy"]
            for friend_enemy in team_list:
                for clusterx in range(self.role_amount):
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_px"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_py"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_psymy"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_unnorm_speed"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_dir_x"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_dir_y"] = 0
                    self.parsed_result[f"{friend_enemy}_{clusterx}_rolecentroid_unnorm_diff"] = 0
            for player, playerdata in frame.players_data.items():
                assert player.player_id in teams
                self._player_stats(player, playerdata, teams, period_start, min_team, max_team, 
                    right, friend, role_count) 

            # add distance between the role centroids
            for friend_enemy in team_list:
                for start_cluster in range(self.role_amount):
                    x1 = self.parsed_result[f"{friend_enemy}_{start_cluster}_rolecentroid_px"]
                    y1 = self.parsed_result[f"{friend_enemy}_{start_cluster}_rolecentroid_py"]
                    for next_cluster in range(start_cluster+1, 4):
                        x2 = self.parsed_result[f"{friend_enemy}_{next_cluster}_rolecentroid_px"]
                        y2 = self.parsed_result[f"{friend_enemy}_{next_cluster}_rolecentroid_py"]
                        self.parsed_result[f"{friend_enemy}_{start_cluster}->{next_cluster}_distx"] = abs(x1 - x2)
                        self.parsed_result[f"{friend_enemy}_{start_cluster}->{next_cluster}_disty"] = abs(y1 - y2)

            _, ball_owner_role, _ = teams[self.ball_dist[1].player_id]
            self.parsed_result["ball_owner_role"] = ball_owner_role
            self.parsed_result["ball_owner_normrole"] = ball_owner_role / 3

            # 0: noone, 1 keeper, 2 defense, 3 mid, 4 attack
            ball_owner_role += 1
            if self.ball_dist[0] > self.ball_owner_cutoff:
                ball_owner_role = 0
            self.parsed_result["ball_owner_v2_role"] = ball_owner_role
            self.parsed_result["ball_owner_v2_normrole"] = ball_owner_role / 4

            _, ball_owner_role, _ = teams[self.ball_dist[1].player_id]
            if self.ball_dist[0] > self.ball_owner_cutoff:
                if len(self.parse_data) > 0:
                    ball_owner_role = self.parse_data[-1]["ball_owner_v3_role"]
                else:
                    ball_owner_role = 3
            self.parsed_result["ball_owner_v3_role"] = ball_owner_role
            self.parsed_result["ball_owner_v3_normrole"] = ball_owner_role / 3
            # keeper is counted as defender
            self.parsed_result["ball_owner_v4_role"] = max(ball_owner_role, 1)
            self.parsed_result["ball_owner_v4_normrole"] = (max(ball_owner_role, 1) - 1) / 2

            self.parsed_result["enemy_actual_dir_x"] /= 11
            self.parsed_result["enemy_actual_dir_y"] /= 11
            self.parsed_result["friend_actual_dir_x"] /= 11
            self.parsed_result["friend_actual_dir_y"] /= 11
            self.parsed_result["actual_dir_x"] /= 22
            self.parsed_result["actual_dir_y"] /= 22

            # span
            self.parsed_result["friend_span_px"] = max_team["friendx"] - min_team["friendx"]
            self.parsed_result["friend_span_py"] = max_team["friendy"] - min_team["friendy"]
            self.parsed_result["enemy_span_px"] = max_team["enemyx"] - min_team["enemyx"]
            self.parsed_result["enemy_span_py"] = max_team["enemyy"] - min_team["enemyy"]

            # min / max
            self.parsed_result["friend_max_px"] = max_team["friendx"]
            self.parsed_result["friend_max_py"] = max_team["friendy"]
            self.parsed_result["enemy_max_px"] = max_team["enemyx"]
            self.parsed_result["enemy_max_py"] = max_team["enemyy"]

            self.parsed_result["friend_min_px"] = min_team["friendx"]
            self.parsed_result["friend_min_py"] = min_team["friendy"]
            self.parsed_result["enemy_min_px"] = min_team["enemyx"]
            self.parsed_result["enemy_min_py"] = min_team["enemyy"]

            px_enemy = []
            for i in range(1, 11): 
                col = f"player{i+1}_enemy_px"
                px_enemy.append(self.parsed_result[col])

            px_enemy = sorted(px_enemy)
            self.parsed_result["enemy_line"] = np.mean(px_enemy[:3])
            if right == friend:
                self.parsed_result["enemy_line_actual"] = 1 - self.parsed_result["enemy_line"]
            else:
                self.parsed_result["enemy_line_actual"] = self.parsed_result["enemy_line"]

            self.parse_data.append(self.parsed_result)
            old_period = frame.period
            frame_index += 1

        # print(skip_counter)
        return self.parse_data


    def animate_roles(self, labels, filename="temp.gif", start_index=0, length=1000, friend="friend_actual", enemy="enemy_actual", folder="temp", name="", speedup=4):
        # First set up the figure, the axis
        pitch = Pitch(pitch_type='metricasports', goal_type='line', pitch_width=68, pitch_length=105)
        fig, ax = pitch.draw(figsize=(8, 5.2))
        ax.set_title(name)

        # then setup the pitch plot markers we want to animate
        marker_kwargs = {'markeredgecolor': 'black', 'linestyle': 'None'}
        # circle for away team
        ball, = ax.plot([], [], ms=6, marker="o", markerfacecolor='w', zorder=3, **marker_kwargs)
        away, = ax.plot([], [], ms=10, marker="o", markerfacecolor='#b94b75', **marker_kwargs)  # red/maroon

        home = []
        for i in range(len(labels)):
            home += ax.plot([], [], "rs", ms=10, marker=f"$ {i} $", color="red", label=labels[i])

        enemy_x = [col for col in self.home_df.columns if enemy + '_x' in col]
        enemy_y = [col for col in self.home_df.columns if enemy + '_y' in col]

        legend = ax.legend(handles=home)

        # animation function
        def animate(i):
            i += start_index
            """ Function to animate the data. Each frame it sets the data for the players and the ball."""
            # set the ball data with the x and y positions for the ith frame
            ball.set_data(self.home_df.loc[i, ["ball_actual_x"]], self.home_df.loc[i, ["ball_actual_y"]])
            away.set_data(self.home_df.loc[i, enemy_x], self.home_df.loc[i, enemy_y])

            home_pos, _ = self.get_current_team_positions(i)
            for role in range(len(labels)):
                players = [elem[0] for elem in home_pos.values() if elem[1] == role]
                cols_x = list(map(lambda x: x + "_actual_x", players))
                cols_y = list(map(lambda x: x + "_actual_y", players))
                home[role].set_data(self.home_df.loc[i, cols_x], self.home_df.loc[i, cols_y])


            return ball, away, home, legend


        # call the animator, animate so 250 ms between frames
        anim = animation.FuncAnimation(fig, animate, frames=length, interval=250, blit=True, repeat=False)
        anim.save(folder + "/" + filename, fps=self.fps * speedup, dpi=80)




# d = TracabDataset(1/6)
# datafilepath = os.path.join(os.path.curdir, "..", "data", "BEL_CRO_2181417.dat")
# metafilepath = os.path.join(os.path.curdir, "..", "data",  "2181417_metadata.xml")
# d.open_dataset(datafilepath, metafilepath)

# print(d.get_home_df().columns)
# print('----------------------')
# print(d.get_home_df().head())
# print('----------------------')
# print(d.get_away_df().head())
