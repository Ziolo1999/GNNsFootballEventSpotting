from statsbombpy import sb
import numpy as np 
from datetime import datetime, timedelta
from dataclasses import dataclass
from matplotlib import pyplot as plt
import os

def main():
    framerate = 5
    total_frames = 50_000

    classes = ['Starting XI', 'Half Start', 'Pass', 'Carry', 'Ball Receipt*',
        'Pressure', 'Foul Won', 'Foul Committed', 'Interception',
        'Injury Stoppage', 'Ball Recovery', 'Duel', 'Substitution',
        'Dribble', 'Miscontrol', 'Block', 'Dribbled Past', 'Clearance',
        'Goal Keeper', 'Dispossessed', 'Bad Behaviour', 'Shot',
        'Tactical Shift', '50/50', 'Referee Ball-Drop', 'Half End', 'Own Goal Against', 
        'Offside','Own Goal For', 'Error', 'Shield', 'Camera On', 'Player Off', 'Player On','Camera off']

    classes_enc = dict(zip(classes, np.arange(len(classes))))

    tournament_enc = {
        43: "EC2020", 
        3: "WC2018", 
        106: "WC2022"
        }

    annotations = {}

    game_names = {}

    related_competitions = [
        (55,43),
        (43,106),
        (43,3)]

    @dataclass
    class Annotations:
        timestamps = {}
        occurences = {}
        avg_length = {}
        frames = {}

    ann = Annotations() 

    for competition, season in related_competitions:
        df_matches = sb.matches(competition_id=competition, season_id=season)
        # Find Belgium games 
        selected_matches = df_matches[(df_matches['home_team']=='Belgium') | (df_matches['away_team']=='Belgium')]
        selected_matches_id = selected_matches.match_id.values
        # Collect the team names 
        teams = selected_matches.loc[:,["home_team", "away_team"]].values.tolist()
        competition_stage = selected_matches.competition_stage.values.tolist()
        game_names[tournament_enc[season]] = teams
        # Store games from particular season
        season_annotations = {}
        for i, match in enumerate(selected_matches_id):
            annotation_matrix = np.zeros((total_frames, len(classes)))
            # Collect events
            event = sb.events(match_id=match)
            event = event.sort_values(by="timestamp").reset_index()
            series = event.type
            # Find indices when events swapped
            start_indices = series.index[series != series.shift(1)].tolist()
            
            for start in range(len(start_indices)-1):
                start_indx = start_indices[start]
                end_indx = start_indices[start+1]
                value = series[start_indx]
                # Get timestamps
                timestamp_tuple = (event.timestamp[start_indx], event.timestamp[end_indx])
                start_seconds = (datetime.strptime(timestamp_tuple[0], '%H:%M:%S.%f')- datetime.strptime('00:00:00.000', '%H:%M:%S.%f')).total_seconds()
                end_seconds = (datetime.strptime(timestamp_tuple[1], '%H:%M:%S.%f') - datetime.strptime('00:00:00.000', '%H:%M:%S.%f')).total_seconds()
                # Collects period information allowing to adjust timstamps for second periods
                period = event.period[start_indx]
                if period == 1:
                    first_period_time = end_seconds
                elif period == 2:
                    start_seconds += first_period_time
                    end_seconds+= first_period_time

                # Adjust to corresponding frame indx 
                frame_tuple = (int(np.floor(start_seconds*framerate)), int(np.ceil(end_seconds*framerate)+framerate))
                annotation_matrix[frame_tuple[0]:frame_tuple[1], classes_enc[value]]=1
            
                # Store statistics
                try:
                    ann.occurences[value] += 1
                    ann.avg_length[value] = (ann.avg_length[value]*(ann.occurences[value]-1) + (end_seconds-start_seconds)) / ann.occurences[value]
                except KeyError:
                    ann.occurences[value] = 1
                    ann.avg_length[value] = end_seconds-start_seconds
            # Get game names
            name_shortcut = f"{teams[i][0][0:3].upper()}-{teams[i][1][0:3].upper()}"
            if "ENG" in name_shortcut:
                name_shortcut = f"{name_shortcut}_{competition_stage[i].replace(' ', '_').lower()}"
            # Store annotations
            season_annotations[name_shortcut] = annotation_matrix
        
        annotations[tournament_enc[season]] = season_annotations

    ann.occurences
    ann.avg_length

    plt.barh(list(ann.occurences.keys()), list(ann.occurences.values()))
    plt.savefig("Code/plots/annotations_occurences.png")

    plt.barh(list(ann.avg_length.keys()), list(ann.avg_length.values()))
    plt.savefig("Code/plots/annotations_avg_length.png")

    path = "football_games"
    for tournament, games in annotations.items():
        for game in games.keys():
            game_path = f"{path}/{tournament}/{game}"

            if not os.path.exists(game_path):
                game_path = f"{path}/{tournament}/{game[4:7]}-{game[0:3]}{game[7:]}"
            
            array = annotations[tournament][game]
            np.savez(f"{game_path}/annotation.npz", array1=array)


if __name__ == "__main__":
    main()