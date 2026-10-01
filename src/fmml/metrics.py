"""Binary detection metrics (attack = positive class)."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def mcc(tp, fp, tn, fn) -> float:
    """Matthews correlation coefficient from confusion counts (0 when undefined). Unlike the F1 of
    the attack class, it does not reward predicting "attack" for everything when attacks dominate."""
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((float(tp) * tn - float(fp) * fn) / den) if den > 0 else 0.0


def binary_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0), "F1": f1_score(y_true, y_pred, zero_division=0),
            "FPR": fp / (fp + tn) if (fp + tn) else 0.0, "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
            "MCC": mcc(tp, fp, tn, fn), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
