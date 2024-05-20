import numpy as np
import seaborn as sns

def segmentation_correlation(segmentation_results):
    corr_matrix = np.corrcoef(segmentation_results)
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
    return heatmap