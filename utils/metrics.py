import numpy as np
import torch
import logging

logger = logging.getLogger('xairesunet_logger')

def dice(pred, true, k=1):
    intersection = np.sum(pred[true==k]) * 2.0
    pred_sum = np.sum(pred)
    true_sum = np.sum(true)
    if pred_sum + true_sum == 0:
        return 1.0  # Both are empty, Dice score is perfect
    dice = intersection / (pred_sum + true_sum)
    return dice

def seg_acc(pred_mask, gt_mask):
    eps = 1e-10
    p = pred_mask.astype(int)
    g = gt_mask

    pvec = np.reshape(p, p.shape[0] * p.shape[1])
    gvec = np.reshape(g, g.shape[0] * g.shape[1])

    TP_idx = np.where((pvec == 1) & (gvec == 1))
    TP = np.shape(TP_idx)[1]

    FP_idx = np.where((pvec == 1) & (gvec == 0))
    FP = np.shape(FP_idx)[1]

    TN_idx = np.where((pvec == 0) & (gvec == 0))
    TN = np.shape(TN_idx)[1]

    FN_idx = np.where((pvec == 0) & (gvec == 1))
    FN = np.shape(FN_idx)[1]

    seg_acc = (TP + TN) / (TP + TN + FP + FN + eps) * 100

    if TP + FP == 0:
        precision = 1.0  # Both predicted and ground truth masks are empty
    else:
        precision = TP / (TP + FP + eps)

    if TP + FN == 0:
        recall = 1.0  # Both predicted and ground truth masks are empty
    else:
        recall = TP / (TP + FN + eps)

    if precision + recall == 0:
        Fscore = 0.0
    else:
        Fscore = 2 * (precision * recall) / (precision + recall + eps)

    if TP + FP + FN == 0:
        Jaccard = 1.0  # Both predicted and ground truth masks are empty
    else:
        Jaccard = TP / (TP + FP + FN + eps)
    
    DSC = dice(pred_mask, gt_mask, k=1)

    return seg_acc, precision, recall, Fscore, Jaccard, DSC

def calculate_metrics(preds, labels):
    try:
        preds = (preds > 0.5).float()
        labels = labels.float()

        metrics_list = []
        total_false_positives = 0
        total_false_negatives = 0

        for i in range(preds.size(0)):
            pred = preds[i].cpu().numpy().squeeze()
            label = labels[i].cpu().numpy().squeeze()

            seg_accuracy, precision, recall, f1_score, jaccard, dice_score = seg_acc(pred, label)
            
            metrics_list.append({
                'IoU': jaccard,
                'Dice': dice_score,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'Segmentation Accuracy': seg_accuracy
            })

            # Counting false positives and false negatives directly
            false_positives = np.sum((pred == 1) & (label == 0))
            false_negatives = np.sum((pred == 0) & (label == 1))
            total_false_positives += false_positives
            total_false_negatives += false_negatives

        metrics_aggregate = {
            'IoU': np.mean([m['IoU'] for m in metrics_list]),
            'Dice': np.mean([m['Dice'] for m in metrics_list]),
            'Precision': np.mean([m['Precision'] for m in metrics_list]),
            'Recall': np.mean([m['Recall'] for m in metrics_list]),
            'F1 Score': np.mean([m['F1 Score'] for m in metrics_list]),
            'Segmentation Accuracy': np.mean([m['Segmentation Accuracy'] for m in metrics_list]),
            'False Positives': total_false_positives,
            'False Negatives': total_false_negatives,
        }

        return metrics_aggregate

    except Exception as e:
        logger.error(f"Error in calculate_metrics: {e}")
        return {
            'IoU': 0,
            'Dice': 0,
            'Precision': 0,
            'Recall': 0,
            'F1 Score': 0,
            'Segmentation Accuracy': 0,
            'False Positives': 0,
            'False Negatives': 0,
        }

