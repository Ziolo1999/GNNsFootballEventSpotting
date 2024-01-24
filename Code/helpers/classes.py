import torch


# Event name to label index fororor SoccerNet-V2
EVENT_DICTIONARY_V2_ALIVE = {"Alive":0, "Dead":1}
INVERSE_EVENT_DICTIONARY_V2_ALIVE = {0: "Alive", 1: "Dead"}

# Values of the K parameters (in seconds) in the context-aware loss
K_V2_ALIVE = torch.FloatTensor([[-50,-60], [-20,-30], [20,30], [50,60]])