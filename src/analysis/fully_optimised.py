import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import random 
from scipy.stats import skewnorm



Ks = [-100,-50,0,50,100]
K_labels = ["K1", "K2", 0, "K3", "K4"]
frames = np.arange(-150, 150)

probs = []
for i, frame in enumerate(frames):
    if (frame>=0) & (frame<50):
        probs.append(1)
    elif (frame<0) & (frame>=-50):
        probs.append(np.random.rand())
    else:
        probs.append(0)

plt.plot(frames, probs)
plt.fill_between(frames, probs, alpha=0.8, label="Fully optimised probabilities")
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
plt.show()