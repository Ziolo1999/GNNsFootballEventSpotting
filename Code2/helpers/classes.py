import torch
import numpy as np 


selected_classes = ['Pressure', 'Foul Committed', 'Ball Recovery', 'Duel', 'Shot', 'Dribble','Clearance', 'Goal Keeper', "Pass", "Dead"]
loss_weights = torch.tensor([2.0, 32.0, 10.0, 17.0, 31.0, 27.0, 25.0, 28.0, 1.0, 1.0]) 

# Event name to label index fororor SoccerNet-V2
EVENT_DICTIONARY_V2_ALIVE = dict(zip(selected_classes, np.arange(len(selected_classes))))
INVERSE_EVENT_DICTIONARY_V2_ALIVE = dict(zip(np.arange(len(selected_classes)), selected_classes))


def get_K_params(chunk_size, selected_ann=None):

    K_vals = [
        (0.2*chunk_size, 0.1*chunk_size),
        (0.6*chunk_size, 0.4*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.6*chunk_size, 0.4*chunk_size),
        (0.6*chunk_size, 0.4*chunk_size),
        (0.6*chunk_size, 0.4*chunk_size),
        (0.6*chunk_size, 0.4*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size)
        ]
    
    if selected_ann:
        K_vals = [K_vals[EVENT_DICTIONARY_V2_ALIVE[selected_ann]]]
    
    K_LIST = []
    for K in range(4):
        # multiplier = -1 if K in range(2) else 1
        if K==0:
            list_param = [-K_vals[ann][0] for ann in range(len(K_vals))]
        elif K==1:
            list_param = [-K_vals[ann][1] for ann in range(len(K_vals))]
        elif K==2:
            list_param = [K_vals[ann][1] for ann in range(len(K_vals))]
        else:
            list_param = [K_vals[ann][0] for ann in range(len(K_vals))]
        K_LIST.append(list_param)

    K_PARAMS = torch.FloatTensor(K_LIST)
    return K_PARAMS


