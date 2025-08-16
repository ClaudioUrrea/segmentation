# evaluation_metrics.py
import numpy as np
from sklearn.metrics import jaccard_score

def compute_miou(predictions, ground_truths, num_classes=2):
    """Compute mean Intersection over Union (mIoU)."""
    ious = []
    for pred, gt in zip(predictions, ground_truths):
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        iou = jaccard_score(gt_flat, pred_flat, average=None)
        ious.append(np.mean(iou))
    return np.mean(ious)

def compute_statistics(miou_scores):
    """Compute stats: mean, std, min, max, quartiles."""
    mean = np.mean(miou_scores)
    std = np.std(miou_scores)
    min_val = np.min(miou_scores)
    q1 = np.percentile(miou_scores, 25)
    median = np.median(miou_scores)
    q3 = np.percentile(miou_scores, 75)
    max_val = np.max(miou_scores)
    return {
        'mean': mean,
        'std_dev': std,
        'min': min_val,
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': max_val
    }