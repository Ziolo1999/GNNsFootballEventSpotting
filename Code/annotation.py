import xml.etree.ElementTree as ET
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
xml_path = "../football_games/Belgium - Azerbaijan.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

@dataclass
class Annotations:
    timestamps = {}
    occurences = {}
    avg_length = {}
    frames = {}

annotations_encodings = {
    'Goal kick OFF':0,
    'Kick off DEF':1, 
    'Transition DEF':2, 
    'Pressing':3, 
    'Build Up':4, 
    'Cross':5, 
    'Rest defense':6, 
    'Throw In DEF':7, 
    'Throw In OFF':8, 
    'Chance':9, 
    'Final third':10, 
    'Corner OFF':11, 
    'Transition OFF':12, 
    'Goal kick DEF':13, 
    'Oppo Cross':14, 
    'Goal':15, 
    'Free Kick OFF':16, 
    'Free Kick DEF':17, 
    'Oppo Chance':18, 
    'Disallowed Goal':19, 
    'Kick off OFF':20, 
    }

ann = Annotations() 
framerate = 5
total_frames = 30_000

def generate_annotations(xml_path, framerate, total_frames):
    # Parse the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    full_game_cntr = 0
    # Iterate through the all instances
    for instance in root.find('ALL_INSTANCES').iter('instance'):
        code = instance.find('code').text
        # Timeframes used for synchronization
        if code == "Full Game":
            if full_game_cntr == 0:
                time_to_kick_off = float(instance.find('start').text) # Time passed to kick-off
                break_start = float(instance.find('end').text) # Timestamp indicating start of the break
            elif full_game_cntr == 1:
                break_end = float(instance.find('start').text) # Timestamp indicating end of the break
            full_game_cntr += 1

        # Phases in the first half
        if full_game_cntr == 1:
            # Timestamps used to calculate frame indices
            start_time_frame = float(instance.find('start').text) - time_to_kick_off
            end_time_frame = float(instance.find('end').text) - time_to_kick_off
            # Timestamps of phases (same as for frames)
            start_time = float(instance.find('start').text) - time_to_kick_off
            end_time = float(instance.find('end').text) - time_to_kick_off
        # Phases in the second half
        elif full_game_cntr == 2:
            # Timestamps used to calculate frame indices
            start_time_frame = float(instance.find('start').text) - time_to_kick_off - (break_end - break_start)
            end_time_frame = float(instance.find('end').text) - time_to_kick_off - (break_end - break_start)
            # Starts from the 45 minute (doesn't take into account the additional time in the first half)
            start_time = 45*60 + float(instance.find('start').text) - break_end 
            end_time = 45*60 + float(instance.find('end').text) - break_end

        # Calculate nr of occurences and the average length of the phase
        try:
            ann.occurences[code] += 1
            ann.avg_length[code] = (ann.avg_length[code]*(ann.occurences[code]-1) + (end_time-start_time)) / ann.occurences[code]
        except KeyError:
            ann.occurences[code] = 1
            ann.avg_length[code] = end_time-start_time

        try: 
            ann.timestamps[code].append((start_time/60, end_time/60))
        except KeyError:
            ann.timestamps[code]=[]
            ann.timestamps[code].append((start_time/60, end_time/60))
        
        # We need to round the timestamp to get the frame indices
        try: 
            ann.frames[code].append((int(np.ceil(start_time_frame)*framerate), int(np.floor(end_time_frame)*framerate)))
        except KeyError:
            ann.frames[code]=[]
            ann.frames[code].append((int(np.ceil(start_time_frame)*framerate), int(np.floor(end_time_frame)*framerate)))

    annotations = np.zeros((total_frames, len(annotations_encodings.keys())))
    for annotation, encoding in annotations_encodings.items():
        indices = ann.frames[annotation]
        for index in indices:
            annotations[index[0]:index[1]+1,encoding] = 1
    return annotations    

annotations = generate_annotations(xml_path,framerate,total_frames)


# Occurence of annotations
plt.bar(ann.occurences.keys(), ann.occurences.values())
plt.show()

# Average length
plt.bar(list(ann.avg_length.keys())[1:], ann.avg_length.values()[1:])
plt.show()
