import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch 
from helpers.loss import ContextAwareLoss
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


def animate_loss_impact(data, save_dir:str):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(0,300)
    line, = ax.plot(x, data[0], color="#ff7f0e", label="Probability")
    ax.set_ylim(0, 1)
    def animate(i):
        line.set_ydata(data[i])
        return line,
    
    anim = FuncAnimation(fig, animate, frames=len(data), interval=500, blit=True)
    
    plt.title("Animation of loss impact on probabilities")
    plt.legend()
    anim.save(save_dir, fps=10)
    return anim

# Function to animate data and timeline
def animate_data_with_timeline(loss, marginal, Ks, save_path):
    # Set up the figure with two subplots: one for the animation, one for the timeline
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 16), gridspec_kw={'height_ratios': [10, 10, 1]})
    
    # Initial plot for the animation
    x_data = np.arange(len(loss[0]))/100
    line1, = ax1.plot(x_data, loss[0], color="#ff7f0e")
    ax1.set_title("Loss behaviour")
    ax1.set_ylim(0, 5)
    ax1.set_xlabel("Probability")
    ax1.set_ylabel("Loss")
    ax1.set_facecolor('lightgray') 
    ax1.grid(True, color='white')

    # Initial plot for the animation
    line2, = ax2.plot(x_data, marginal[0], color="#ff7f0e")
    ax2.set_title("Marginal loss")
    ax2.set_ylim(-10, 0)
    ax2.set_xlabel("Probability")
    ax2.set_ylabel("ΔLoss")
    ax2.set_facecolor('lightgray') 
    ax2.grid(True, color='white')

    # Add ticks to the timeline subplot
    ax3.set_title("Distance to the event")
    ax3.set_xlim(-150, 150)
    ax3.set_ylim(0, 0.001)
    ax3.set_xticks(Ks)
    ax3.set_yticks([])
    ax3.axis('on') 
    ax3.vlines(Ks, 0, 1, color=mcolors.CSS4_COLORS["darkslategrey"])
    
    for x, label in zip(Ks, ["K1", "K2", "0", "K3", "K5"]):
        plt.text(x, -0.15, label, ha='center', va='top', color='black')
    
    # Progress bar rectangle
    progress_bar = Rectangle((-150, 0), 0, 1, color=mcolors.CSS4_COLORS["seagreen"], alpha=0.5)
    ax3.add_patch(progress_bar)
    
    # Animation update function
    def animate(i):
        # Update the main plot
        line1.set_ydata(loss[i])
        line2.set_ydata(marginal[i])
        # Update the progress bar's width to the current frame
        progress_bar.set_width(i)
        return line1, progress_bar

    # Create the animation
    anim = FuncAnimation(fig, animate, frames=len(loss), interval=200, blit=True)

    # Save the animation
    anim.save(save_path, fps=10)

# LOSS FUNCTIONS
# First function: 
def func1(p):
    return -np.log(1 - p)

# Second function:
def func2(K2, K1, s, p):
    return -np.log(1 - ((K2 - s) / (K2 - K1)) * p)

# Third function: just a zero placeholder, assuming a function that always returns 0
def func3():
    return 0

# Fourth function: 
def func4(K3, s, p):
    return -np.log((s / K3) + ((K3 - s) / K3) * p)

# Fifth function: 
def func5(K4, K3, s, p, ):
    return -np.log(1 - ((s - K3) / (K4 - K3)) * p)

# Sixth function: same as the first one, added here for completeness
def func6(p):
    return -np.log(1 - p)

# DERIVIATIVES OF LOSS FUNCTIONS
def dfunc1(p):
    return -1 / (1 - p)

def dfunc2(K2, K1, s, p):
    return -(K2 - s) / (-K2*(p - 1) - K1 + p*s)

def dfunc3():
    return 0

def dfunc4(K3, s, p):
    return (s - K3) / (K3*p - p*s + s)

def dfunc5(K4, K3, s, p):
    return -(K3 - s) / (-K3*(p - 1) - K4 + p*s)

def dfunc6(p):
    return -1 / (1 - p)




def main():
    parser = argparse.ArgumentParser(prog="loss_visualiser", description="Visualise the Context Aware Loss")
    parser.add_argument("-o", "--output", help="The path to the folder to save visualisations")
    args = parser.parse_args()

    ################################################################
    #                 Animation of loss behaviour                  #
    ################################################################ 
    Ks = np.array([-100, -50, 0, 50, 100])
    probs = np.arange(0.01,1,0.01)
    loss_functions = []
    marginal_loss_reduction = []

    for K in np.arange(-150,150):
        if K <= Ks[0]:
            loss_functions.append([func1(p) for p in probs])
            marginal_loss_reduction.append([dfunc1(p) for p in probs])
        elif Ks[0] < K <= Ks[1]:
            loss_functions.append([func2(Ks[1], Ks[0], K, p) for p in probs])
            marginal_loss_reduction.append([dfunc2(Ks[1], Ks[0], K, p) for p in probs])
        elif Ks[1] < K < Ks[2]:
            loss_functions.append([func3() for p in probs])
            marginal_loss_reduction.append([dfunc3() for p in probs])
        elif Ks[2] <= K < Ks[3]:
            loss_functions.append([func4(Ks[3], K, p) for p in probs])
            marginal_loss_reduction.append([dfunc4(Ks[3], K, p) for p in probs])
        elif Ks[3] <= K < Ks[4]:
            loss_functions.append([func5(Ks[4], Ks[3], K, p) for p in probs])
            marginal_loss_reduction.append([dfunc5(Ks[4], Ks[3], K, p) for p in probs])
        else:
            loss_functions.append([func6(p) for p in probs])
            marginal_loss_reduction.append([dfunc6(p) for p in probs])
    
    animate_data_with_timeline(loss_functions, marginal_loss_reduction, Ks, f'{args.output}/loss_behaviour.mp4')

    
    ################################################################
    #            Animation of loss impact on probabilities         #
    ################################################################                            
    
    
    # Create artificial features
    features = torch.tensor(np.random.random((300)), dtype=torch.float32)
    # List to store outputs
    outputs = []

    # Create basic model 
    class BasicModel(nn.Module):
        def __init__(self, num_classes=300, args=None):
            super(BasicModel, self).__init__()
            self.linear1 = nn.Linear(in_features=num_classes, out_features=num_classes)
            self.linear2 = nn.Linear(in_features=num_classes, out_features=num_classes)
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = F.sigmoid(x)
            return x.reshape((1,300,1))
    
    # Settings
    K = torch.tensor(np.array([[-100, -50, 50, 100]]).T)
    labels = torch.tensor(np.arange(-150,150).reshape(1,300,1))
    model = BasicModel()
    criterion = ContextAwareLoss(K=K)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, 
                                betas=(0.9, 0.999), eps=1e-07, 
                                weight_decay=0, amsgrad=False)
    # Train
    for _  in range(100):
        output = model(features)
        loss = criterion(labels, output, device) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.append(list(1-output[0,:,0].detach().numpy()))
    
    animate_loss_impact(outputs, f'{args.output}/loss_impact_probs.mp4')



if __name__ == '__main__':
    main()
