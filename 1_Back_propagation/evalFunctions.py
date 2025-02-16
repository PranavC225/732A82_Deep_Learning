import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    correct = np.sum(LPred == LTrue)
    total = len(LTrue)
    acc = correct/total
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    unique_labels = np.unique(np.concatenate((LPred, LTrue)))
    num_classes = len(unique_labels)
    cM = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(LTrue, LPred):
        cM[pred_label, true_label] += 1
    
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    correct_predictions = np.trace(cM)
    total_predictions = np.sum(cM)
    acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return acc
