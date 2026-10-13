"""
models/qsvm_model.py
--------------------
Quantum Support Vector Machine (QSVM / QSVC) for Fraud Detection.
Uses FidelityQuantumKernel with ZZFeatureMap(feature_dimension=4, reps=2).
Matches Slide 7 and Slide 5 of the presentation.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QuantumFraudQSVM:
    """
    Quantum Support Vector Classifier (QSVC) using FidelityQuantumKernel
    and ZZFeatureMap as specified in the presentation.
    """
    def __init__(self, feature_dimension: int = 4, reps: int = 2):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.backend_name = "Local Quantum Simulator (Statevector)"
        self.feature_map_name = f"ZZFeatureMap (dim={feature_dimension}, reps={reps})"
        self.model_name = "Quantum Support Vector Machine (QSVM)"
        self.is_trained = False
        
        # Build ZZFeatureMap matching Slide 7:
        # fmap = ZZFeatureMap(feature_dimension=4, reps=2)
        # qk = FidelityQuantumKernel(feature_map=fmap)
        # qsvc = QSVC(quantum_kernel=qk)
        self.feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=reps, entanglement="linear")
        self.quantum_kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits the QSVC model on the training samples."""
        print(f"[QSVM] Training QSVC on {len(X_train)} samples with {self.feature_dimension} qubits...")
        self.qsvc.fit(X_train, y_train)
        self.is_trained = True
        print(f"[QSVM] Training complete! Number of support vectors: {len(self.qsvc.support_)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("QSVM model is not trained yet.")
        return self.qsvc.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("QSVM model is not trained yet.")
        return self.qsvc.decision_function(X)

    def get_fraud_risk_score(self, x_single: np.ndarray) -> float:
        """
        Computes calibrated Fraud Risk Score from SVM decision distance.
        Sigmoid activation over decision_function value: 1 / (1 + exp(-decision_value)).
        """
        if x_single.ndim == 1:
            x_single = x_single.reshape(1, -1)
        dec = float(self.decision_function(x_single)[0])
        # Sigmoid calibration: score > 0 -> Fraud (label 1), score < 0 -> Legit (label 0)
        prob = 1.0 / (1.0 + np.exp(-1.8 * dec))
        return round(float(prob * 100.0), 1)

    def save(self, model_dir: str = "models"):
        os.makedirs(model_dir, exist_ok=True)
        # Save QSVC using dill or joblib
        dill_path = os.path.join(model_dir, "quantum_qsvm.joblib")
        joblib.dump(self.qsvc, dill_path)
        
        meta = {
            "model_name": self.model_name,
            "feature_dimension": self.feature_dimension,
            "reps": self.reps,
            "backend_name": self.backend_name,
            "feature_map_name": self.feature_map_name,
            "n_support_vectors": int(len(self.qsvc.support_)) if self.is_trained else 0
        }
        with open(os.path.join(model_dir, "qsvm_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[QSVM] Saved model to '{model_dir}/'.")

    @classmethod
    def load(cls, model_dir: str = "models"):
        dill_path = os.path.join(model_dir, "quantum_qsvm.joblib")
        meta_path = os.path.join(model_dir, "qsvm_meta.json")
        
        if not os.path.exists(dill_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"QSVM files missing in {model_dir}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        obj = cls(
            feature_dimension=meta.get("feature_dimension", 4),
            reps=meta.get("reps", 2)
        )
        obj.qsvc = joblib.load(dill_path)
        obj.is_trained = True
        return obj
