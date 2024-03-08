import torch
import numpy as np 

# selected_classes = ['Pass', 'Carry', 'Ball Receipt*',
#         'Pressure', 'Foul Won', 'Foul Committed', 'Interception',
#         'Ball Recovery', 'Duel', 'Substitution',
#         'Dribble', 'Miscontrol', 'Block', 'Dribbled Past', 'Clearance',
#         'Goal Keeper', 'Dispossessed','Shot']

# selected_classes = ['None','Pressure', 'Foul Won', 'Foul Committed', 'Interception',
#         'Ball Recovery', 'Duel', 'Substitution',
#         'Dribble', 'Miscontrol', 'Block', 'Dribbled Past', 'Clearance',
#         'Goal Keeper', 'Dispossessed','Shot']

selected_classes = ['Pressure', 'Foul Committed', 'Ball Recovery', 'Duel', 'Shot', 'Dribble','Clearance', 'Goal Keeper', "Pass", "Dead"]

# Event name to label index fororor SoccerNet-V2
EVENT_DICTIONARY_V2_ALIVE = dict(zip(selected_classes, np.arange(len(selected_classes))))
INVERSE_EVENT_DICTIONARY_V2_ALIVE = dict(zip(np.arange(len(selected_classes)), selected_classes))

# Values of the K parameters (in seconds) in the context-aware loss
# K_V2_ALIVE = torch.FloatTensor([
#     [-40,-5,-5,-10,-30,-20,-20,-20,-30,-20,-20,-10,-10,-20,-30,-20,-30,-30,-30], 
#     [-20,-3,-3,-5,-15,-10,-10,-10,-15,-10, -10,-5,-5,-10,-15,-10,-15,-15,-15], 
#     [20,3,3,5,15,10,10,10,15,10,10,5,5,10,15,10,15,15,15], 
#     [40,5,5,10,30,20,20,20,30,20,20,10,10,20,30,20,30,30,30]]
#     )

# K_V2_ALIVE = torch.FloatTensor([
#     [-40,-30,-20,-20,-20,-30,-20,-20,-10,-10,-20,-30,-20,-30,-30,-30], 
#     [-20,-15,-10,-10,-10,-15,-10, -10,-5,-5,-10,-15,-10,-15,-15,-15], 
#     [20,15,10,10,10,15,10,10,5,5,10,15,10,15,15,15], 
#     [40,30,20,20,20,30,20,20,10,10,20,30,20,30,30,30]
#     ])

K_V2_ALIVE = torch.FloatTensor([
    [-10,-20,-20,-10,-20,-10,-10,-20], 
    [-5,-10,-10,-5,-10,-5,-5,-10], 
    [5,10,10,5,10,5,5,10], 
    [10,20,20,10,20,10,10,20]
    ])

def get_K_params(chunk_size):

    K_vals = [
        (0.2*chunk_size, 0.1*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size),
        (0.4*chunk_size, 0.2*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size),
        (0.2*chunk_size, 0.1*chunk_size)
        ]
    
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