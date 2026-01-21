"""
Evaluation metrics for scoring tasks.

Includes:
- Quadratic Weighted Kappa (QWK) - primary metric for ASAP
- Linear Weighted Kappa
- Accuracy
- Confusion Matrix
"""

import numpy as np
from typing import List, Optional, Tuple, Union


def confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        min_rating: Minimum rating value (auto-detected if None)
        max_rating: Maximum rating value (auto-detected if None)

    Returns:
        Confusion matrix of shape (num_ratings, num_ratings)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=np.int64)

    for true, pred in zip(y_true, y_pred):
        conf_mat[int(true - min_rating), int(pred - min_rating)] += 1

    return conf_mat


def quadratic_weighted_kappa(
    y_true: List[int],
    y_pred: List[int],
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK is the primary evaluation metric for ASAP essay scoring.
    It measures agreement between two raters, accounting for chance agreement
    and penalizing disagreements proportionally to their squared distance.

    Interpretation:
        < 0: Less than chance agreement
        0.01-0.20: Slight agreement
        0.21-0.40: Fair agreement
        0.41-0.60: Moderate agreement
        0.61-0.80: Substantial agreement
        0.81-1.00: Almost perfect agreement

    Args:
        y_true: True labels (human scores)
        y_pred: Predicted labels (model scores)
        min_rating: Minimum rating value (auto-detected if None)
        max_rating: Maximum rating value (auto-detected if None)

    Returns:
        QWK score in range [-1, 1], higher is better

    Example:
        >>> y_true = [0, 1, 2, 3, 2, 1]
        >>> y_pred = [0, 1, 2, 2, 2, 1]
        >>> qwk = quadratic_weighted_kappa(y_true, y_pred)
        >>> print(f"QWK: {qwk:.4f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return 0.0

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = int(max_rating - min_rating + 1)

    # Build confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, min_rating, max_rating)

    # Build weight matrix (quadratic weights)
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    # Normalize confusion matrix
    conf_mat = conf_mat.astype(np.float64)
    n = conf_mat.sum()
    if n == 0:
        return 0.0

    # Expected matrix (outer product of marginals)
    sum0 = conf_mat.sum(axis=0)
    sum1 = conf_mat.sum(axis=1)
    expected = np.outer(sum1, sum0) / n

    # Compute kappa
    observed_weighted = (weights * conf_mat).sum()
    expected_weighted = (weights * expected).sum()

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    kappa = 1.0 - (observed_weighted / expected_weighted)
    return float(kappa)


def linear_weighted_kappa(
    y_true: List[int],
    y_pred: List[int],
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
) -> float:
    """
    Compute Linear Weighted Kappa.

    Similar to QWK but uses linear weights instead of quadratic.
    Less commonly used for essay scoring but can be useful for
    ordinal data with equal intervals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        min_rating: Minimum rating value
        max_rating: Maximum rating value

    Returns:
        Linear weighted kappa score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    if len(y_true) == 0:
        return 0.0

    if min_rating is None:
        min_rating = min(y_true.min(), y_pred.min())
    if max_rating is None:
        max_rating = max(y_true.max(), y_pred.max())

    num_ratings = int(max_rating - min_rating + 1)

    # Build confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, min_rating, max_rating)

    # Build weight matrix (linear weights)
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = abs(i - j) / (num_ratings - 1) if num_ratings > 1 else 0

    # Normalize confusion matrix
    conf_mat = conf_mat.astype(np.float64)
    n = conf_mat.sum()
    if n == 0:
        return 0.0

    # Expected matrix
    sum0 = conf_mat.sum(axis=0)
    sum1 = conf_mat.sum(axis=1)
    expected = np.outer(sum1, sum0) / n

    # Compute kappa
    observed_weighted = (weights * conf_mat).sum()
    expected_weighted = (weights * expected).sum()

    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0

    kappa = 1.0 - (observed_weighted / expected_weighted)
    return float(kappa)


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute simple accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score in range [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(y_true == y_pred))


def mean_absolute_error(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        MAE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        RMSE score (lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def adjacent_agreement(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute adjacent agreement rate.

    Measures the proportion of predictions that are within 1 point
    of the true score. Useful for ordinal data.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Adjacent agreement rate in range [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def compute_all_metrics(
    y_true: List[int],
    y_pred: List[int],
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
) -> dict:
    """
    Compute all available metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        min_rating: Minimum rating value
        max_rating: Maximum rating value

    Returns:
        Dictionary with all metric values
    """
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, min_rating, max_rating),
        "lwk": linear_weighted_kappa(y_true, y_pred, min_rating, max_rating),
        "accuracy": accuracy(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "adjacent_agreement": adjacent_agreement(y_true, y_pred),
        "n_samples": len(y_true),
    }


def human_agreement_qwk(
    score1: List[int],
    score2: List[int],
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
) -> float:
    """
    Compute QWK between two human raters.

    This serves as an upper bound for model performance -
    a model achieving human-level QWK is performing as well
    as one human rater compared to another.

    Args:
        score1: First rater's scores
        score2: Second rater's scores
        min_rating: Minimum rating value
        max_rating: Maximum rating value

    Returns:
        QWK between human raters
    """
    return quadratic_weighted_kappa(score1, score2, min_rating, max_rating)
