# visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_segmentation(img, pred, gt=None, save_path=None):
    """Plot image, prediction, and ground truth."""
    fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[1].imshow(pred, cmap='gray')
    axes[1].set_title('Prediction')
    if gt is not None:
        axes[2].imshow(gt, cmap='gray')
        axes[2].set_title('Ground Truth')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_boxplots(stats_dict, title='Performance Boxplots', save_path=None):
    """Plot boxplots from statistics."""
    data = []
    labels = []
    for key, stats in stats_dict.items():
        labels.append(key)
        data.append(stats['miou_scores'])  # Assume stats has list of miou_scores
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()