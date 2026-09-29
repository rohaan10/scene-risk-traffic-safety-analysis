"""
Evaluation Metrics
──────────────────
Statistical functions for comparing VLM risk predictions against
human-annotated ground truth scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)


@dataclass
class CorrelationResult:
    """Correlation metrics between predicted and ground-truth scores."""

    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    mae: float              # Mean Absolute Error
    rmse: float             # Root Mean Squared Error
    n: int                  # sample size


@dataclass
class ClassificationResult:
    """Binary classification metrics (Hazard / Non-Hazard)."""

    threshold: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion: np.ndarray   # 2×2 confusion matrix
    n: int


# ═════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════

def compute_correlation(
    predictions: List[float] | np.ndarray,
    ground_truth: List[float] | np.ndarray,
) -> CorrelationResult:
    """
    Compute Pearson *r*, Spearman *ρ*, MAE, and RMSE between
    *predictions* and *ground_truth* score arrays.

    Parameters
    ----------
    predictions : array-like of float
        Model-predicted risk scores (0–100).
    ground_truth : array-like of float
        Human-annotated risk scores (0–100).

    Returns
    -------
    CorrelationResult
    """
    preds = np.asarray(predictions, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)

    if len(preds) != len(gt):
        raise ValueError(
            f"Length mismatch: predictions ({len(preds)}) vs "
            f"ground_truth ({len(gt)})"
        )

    if len(preds) < 3:
        raise ValueError("Need at least 3 samples for meaningful correlation.")

    r, r_p = pearsonr(preds, gt)
    rho, rho_p = spearmanr(preds, gt)
    mae = float(mean_absolute_error(gt, preds))
    rmse = float(np.sqrt(mean_squared_error(gt, preds)))

    return CorrelationResult(
        pearson_r=float(r),
        pearson_p=float(r_p),
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        mae=mae,
        rmse=rmse,
        n=len(preds),
    )


def compute_classification_metrics(
    predictions: List[float] | np.ndarray,
    ground_truth: List[float] | np.ndarray,
    threshold: int = 70,
) -> ClassificationResult:
    """
    Binarise scores at *threshold* (≥ threshold → **Hazard**) and compute
    Precision, Recall, F1, and Accuracy.

    Parameters
    ----------
    predictions : array-like of float
        Model-predicted risk scores (0–100).
    ground_truth : array-like of float
        Human-annotated risk scores (0–100).
    threshold : int
        Score at or above which a scene is labelled *Hazard*.

    Returns
    -------
    ClassificationResult
    """
    preds = np.asarray(predictions, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)

    if len(preds) != len(gt):
        raise ValueError(
            f"Length mismatch: predictions ({len(preds)}) vs "
            f"ground_truth ({len(gt)})"
        )

    pred_labels = (preds >= threshold).astype(int)
    gt_labels = (gt >= threshold).astype(int)

    return ClassificationResult(
        threshold=threshold,
        precision=float(precision_score(gt_labels, pred_labels, zero_division=0)),
        recall=float(recall_score(gt_labels, pred_labels, zero_division=0)),
        f1=float(f1_score(gt_labels, pred_labels, zero_division=0)),
        accuracy=float(accuracy_score(gt_labels, pred_labels)),
        confusion=confusion_matrix(gt_labels, pred_labels, labels=[0, 1]),
        n=len(preds),
    )


# ── Pretty-printing helpers ──────────────────────────────────────────────

def format_correlation(result: CorrelationResult, label: str = "") -> str:
    """Return a human-readable summary string."""
    tag = f" [{label}]" if label else ""
    return (
        f"Correlation Metrics{tag} (n={result.n})\n"
        f"  Pearson r   = {result.pearson_r:+.4f}  (p = {result.pearson_p:.2e})\n"
        f"  Spearman ρ  = {result.spearman_rho:+.4f}  (p = {result.spearman_p:.2e})\n"
        f"  MAE         = {result.mae:.2f}\n"
        f"  RMSE        = {result.rmse:.2f}"
    )


def format_classification(result: ClassificationResult, label: str = "") -> str:
    """Return a human-readable summary string."""
    tag = f" [{label}]" if label else ""
    tn, fp, fn, tp = result.confusion.ravel()
    return (
        f"Classification Metrics{tag} (n={result.n}, threshold={result.threshold})\n"
        f"  Precision   = {result.precision:.4f}\n"
        f"  Recall      = {result.recall:.4f}\n"
        f"  F1 Score    = {result.f1:.4f}\n"
        f"  Accuracy    = {result.accuracy:.4f}\n"
        f"  Confusion   = TP={tp}  FP={fp}  FN={fn}  TN={tn}"
    )
