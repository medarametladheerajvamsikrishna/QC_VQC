"""
models/classical_model.py
-------------------------
Classical Machine Learning Baseline (Random Forest & Logistic Regression)
for financial fraud detection comparison against Quantum ML.
"""

import os
import sys
# Ensure workspace root is in sys.path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

class ClassicalFraudModel:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Random Forest with class weighting to handle class imbalance
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        self.lr = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=random_state
        )
        # Classical SVM matching Slide 8 of presentation
        self.svm = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains Random Forest, Logistic Regression, and Classical SVM baselines."""
        print("[Classical Model] Training Random Forest Classifier...")
        self.rf.fit(X_train, y_train)
        
        print("[Classical Model] Training Classical SVM Classifier...")
        self.svm.fit(X_train, y_train)

        print("[Classical Model] Training Logistic Regression Classifier...")
        self.lr.fit(X_train, y_train)
        
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Calculates evaluation metrics on unseen test set."""
        y_pred = self.rf.predict(X_test)
        y_prob = self.rf.predict_proba(X_test)[:, 1]

        metrics = {
            "model_name": "Random Forest (Classical)",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred,
            "y_prob": y_prob
        }
    def evaluate_svm(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Calculates evaluation metrics for Classical SVM baseline (Slide 8)."""
        y_pred = self.svm.predict(X_test)
        y_prob = self.svm.predict_proba(X_test)[:, 1]

        return {
            "model_name": "Classical SVM",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred,
            "y_prob": y_prob
        }

    def save(self, filepath: str = "models/random_forest.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.rf, filepath)
        print(f"[Classical Model] Saved trained Random Forest to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str = "models/random_forest.joblib"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        model_obj = cls()
        model_obj.rf = joblib.load(filepath)
        model_obj.is_trained = True
        return model_obj

if __name__ == "__main__":
    from preprocessing.preprocessing import load_and_clean_data, prepare_data

    df = load_and_clean_data()
    data = prepare_data(df)
    
    baseline = ClassicalFraudModel()
    baseline.train(data["X_train_full"], data["y_train"])
    results = baseline.evaluate(data["X_test_full"], data["y_test"])
    
    print("\n--- Classical Model Test Evaluation ---")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1 Score : {results['f1']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    
    baseline.save()
