"""
evaluation/evaluate.py
----------------------
Model evaluation suite for Classical vs Quantum fraud detection models.
Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
Saves comparison table and visualization artifacts.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def compute_model_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> dict:
    """
    Computes standard academic evaluation metrics.
    Emphasizes Recall and F1-score due to fraud class imbalance.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Confusion_Matrix": cm
    }

def generate_comparison_table(metrics_list: list, output_csv: str = "results/metrics.csv") -> pd.DataFrame:
    """
    Generates and saves the Classical vs Quantum metrics comparison table.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []
    for m in metrics_list:
        rows.append({
            "Metric": m["Model"],
            "Accuracy": f"{m['Accuracy']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall": f"{m['Recall']:.4f}",
            "F1 Score": f"{m['F1 Score']:.4f}",
            "ROC-AUC": f"{m['ROC-AUC']:.4f}",
            "TN": m["TN"],
            "FP": m["FP"],
            "FN": m["FN"],
            "TP": m["TP"]
        })
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(output_csv, index=False)
    print(f"[Evaluation] Comparison metrics saved to '{output_csv}'.")
    return df_comp

def plot_side_by_side_confusion_matrices(
    classical_cm: np.ndarray,
    quantum_cm: np.ndarray,
    output_png: str = "results/confusion_matrix.png"
):
    """
    Plots side-by-side confusion matrices for Random Forest vs Quantum VQC.
    Clearly marks TN, FP, FN, and TP.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = [["TN", "FP"], ["FN", "TP"]]

    # Classical Plot
    sns.heatmap(
        classical_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["Legitimate (0)", "Fraudulent (1)"],
        yticklabels=["Legitimate (0)", "Fraudulent (1)"]
    )
    axes[0].set_title("Classical Baseline (Random Forest)\nConfusion Matrix", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("Actual Label", fontsize=10)

    # Quantum Plot
    sns.heatmap(
        quantum_cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=axes[1],
        xticklabels=["Legitimate (0)", "Fraudulent (1)"],
        yticklabels=["Legitimate (0)", "Fraudulent (1)"]
    )
    axes[1].set_title("Quantum AI (VQC 4-Qubit)\nConfusion Matrix", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("Actual Label", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"[Evaluation] Side-by-side confusion matrix plot saved to '{output_png}'.")

def plot_roc_curves(
    y_true: np.ndarray,
    models_probs: dict,
    output_png: str = "results/roc_curves.png"
):
    """
    Plots ROC curves for multiple models on the same test set.
    models_probs: dict of {model_label: y_prob}
    """
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
    
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728"]
    for i, (name, prob) in enumerate(models_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[Evaluation] ROC curve plot saved to '{output_png}'.")

