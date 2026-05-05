import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary classification accuracy from predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def binary_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute ROC-AUC, returning None when it is undefined."""
    y_true_array = np.asarray(y_true)
    if np.unique(y_true_array).size < 2:
        return None
    return float(roc_auc_score(y_true_array, y_prob))


def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Return standard binary classification metrics."""
    return {
        "accuracy": binary_accuracy(y_true, y_prob, threshold=threshold),
        "roc_auc": binary_roc_auc(y_true, y_prob),
    }
