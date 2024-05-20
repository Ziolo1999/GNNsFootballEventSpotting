import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import random 
from scipy.stats import skewnorm



# Optimization to find the scale for which the peak is approximately 1
def scale_distribution(skew_distribution, loc):
    mode = np.argmax(skew_distribution)
    shift = mode - loc
    slope = skew_distribution[-1] - skew_distribution[-2] # For extrapolation
    # Roll the array to place the distribution on the centre
    rolled_skew_distribution = np.roll(skew_distribution, -shift)
    # Extrapolate the distribution to manage tail events 
    for i in range(1, shift + 1):
        rolled_skew_distribution[-i] = rolled_skew_distribution[-(i+1)] + slope
    # The peak reaches for probability
    rolled_skew_distribution /= np.max(rolled_skew_distribution)
    return rolled_skew_distribution

def generate_artificial_predictions():
    K_param = 100
    sigma = K_param/4
    scaler = K_param * 5/8
    frames = np.arange(-150, 150)
    normal_distribution = norm.pdf(frames, 0, sigma) * scaler
    noise = np.random.normal(0, 0.05, 300)
    noise_distribution = np.min((np.max((normal_distribution + noise, np.zeros(300)), axis=0), np.ones(300)), axis=0)
    
    for i in np.arange(100,150):
        noise_distribution[i] = random.random()
    
    # Generates_skewed targets
    a = 4
    skew_distribution = skewnorm.pdf(frames, a, loc=0, scale=1.5*sigma)
    skew_distribution = scale_distribution(skew_distribution, 150)
    # Skewed noise distribution
    skewed_noise_distribution = np.min((np.max((skew_distribution + noise, np.zeros(300)), axis=0), np.ones(300)), axis=0)
    for i in np.arange(100,150):
        skewed_noise_distribution[i] = random.random()

    return normal_distribution, noise_distribution, skew_distribution, skewed_noise_distribution, frames

Ks = [-100,-50,0,50,100]
K_labels = ["K1", "K2", 0, "K3", "K4"]

def draw_artificial_predictions(frames, noise_distribution, save_dir=None):
    plt.plot(frames, noise_distribution)
    plt.fill_between(frames, noise_distribution, alpha=0.8, label="Artificial Predictions")
    plt.vlines(Ks, ymin= -0.1, ymax=1.1)
    
    for x, label in zip(Ks, K_labels):
        plt.text(x, -0.15, label, ha='center', va='top', color='black')
        
    plt.text(-125, 0.6, r'$-\ln(1 - p)$', fontsize=15,ha='center')
    plt.text(-75, 0.8, r'$-\ln \left( 1 - \frac{K_{2}^{c} - s}{K_{2}^{c} - K_{1}^{c}} p \right)$', fontsize=15, ha='center')
    plt.text(-25, 1.2, r'$0$', fontsize=15, ha='center')
    plt.text(25, 1.2, r'$-\ln \left( \frac{s}{K_{3}^{c}} + \frac{K_{3}^{c} - s}{K_{3}^{c}} p \right)$', fontsize=15, ha='center')
    plt.text(75, 0.8, r'$-\ln \left( 1 - \frac{s - K_{3}^{c}}{K_{4}^{c} - K_{3}^{c}} p \right)$', fontsize=15, ha='center')
    plt.text(125, 0.6, r'$-\ln(1 - p)$', fontsize=15,ha='center')

    plt.ylim(-0.1,1.1)
    plt.legend()
    
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

def draw_artificial_targets(frames, noise_distribution, normal_distribution, save_dir=None, Ks=Ks):
    plt.plot(frames, noise_distribution, label="Artificial Predictions")
    plt.plot(frames, normal_distribution, label="Artificial Targets")
    
    plt.fill_between(frames, noise_distribution, alpha=0.8)
    plt.fill_between(frames, normal_distribution, alpha=0.4)
    
    plt.vlines(Ks, ymin= -0.1, ymax=1.1, )

    for x, label in zip(Ks, K_labels):
        plt.text(x, -0.15, label, ha='center', va='top', color='black')

    plt.ylim(-0.1,1.1)
    plt.legend()
    
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

def draw_areas(frames, noise_distribution, normal_distribution, area_type, save_dir=None):
    plt.plot(frames, normal_distribution, label="Artificial Targets")
    plt.plot(frames, noise_distribution, label="Artificial Predictions")
    plt.vlines(Ks, ymin= -0.1, ymax=1.1)

    if area_type == "tp": 
        plt.fill_between(frames, np.minimum(normal_distribution, noise_distribution), alpha=0.4, label='True Positives', color="#ff7f0e")
    elif area_type == "fp":
        plt.fill_between(frames, normal_distribution, noise_distribution, where=normal_distribution<noise_distribution, alpha=0.4, label='False Positives', color="#ff7f0e")
    elif area_type == "fn":
        plt.fill_between(frames,normal_distribution, noise_distribution, where=normal_distribution>noise_distribution, alpha=0.4, label='False Negatives', color="#1f77b4")
    for x, label in zip(Ks, K_labels):
        plt.text(x, -0.15, label, ha='center', va='top', color='black')
    
    plt.legend()
    plt.ylim(-0.1,1.1)
    
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(prog="metric_visualiser", description="Visualise proposed metric")
    parser.add_argument("-o", "--output", help="The path to the folder to save visualisations")
    args = parser.parse_args()

    normal_distribution, noise_distribution, skew_distribution, skew_noise_distribution, frames = generate_artificial_predictions()
    draw_artificial_predictions(frames, noise_distribution, save_dir=f'{args.output}/artificial_predictions.png')
    draw_artificial_targets(frames, noise_distribution, normal_distribution, save_dir=f'{args.output}/artificial_targets.png')
    draw_artificial_targets(frames, skew_noise_distribution, skew_distribution, save_dir=f'{args.output}/artificial_skewed_targets.png', Ks=[-100,-50,0,75,120])
    draw_areas(frames, noise_distribution, normal_distribution, "fn", save_dir=f'{args.output}/false_negatives.png')
    draw_areas(frames, noise_distribution, normal_distribution, "fp", save_dir=f'{args.output}/false_positives.png')
    draw_areas(frames, noise_distribution, normal_distribution, "tp", save_dir=f'{args.output}/true_positives.png')

if __name__ == "__main__":
    main()