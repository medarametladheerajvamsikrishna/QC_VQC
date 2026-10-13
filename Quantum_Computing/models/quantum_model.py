"""
models/quantum_model.py
-----------------------
Variational Quantum Classifier (VQC) for Financial Fraud Detection.
Genuine QML model using Qiskit Machine Learning and Qiskit 2.x primitives.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
# Use core Qiskit StatevectorSampler — does NOT require qiskit-aer
# This avoids 30+ minute C++ compilation on Streamlit Cloud
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
try:
    from qiskit_algorithms.optimizers import COBYLA, SPSA
except ImportError:
    from qiskit.algorithms.optimizers import COBYLA, SPSA

try:
    from quantum.encoding import build_angle_encoding_circuit
    from quantum.circuit import build_variational_ansatz, build_full_quantum_circuit
except (ModuleNotFoundError, ImportError):
    from encoding import build_angle_encoding_circuit
    from circuit import build_variational_ansatz, build_full_quantum_circuit

class QuantumFraudVQC:
    """
    Genuine Variational Quantum Classifier (VQC) for Fraud Detection.
    - 4 Qubits
    - Angle Feature Encoding (Ry rotations)
    - Parameterized Variational Ansatz with CNOT Entanglement
    - Statevector Local Quantum Simulator
    - Classical Optimizer: COBYLA
    """
    def __init__(
        self,
        num_qubits: int = 4,
        reps: int = 1,
        entanglement: str = "linear",
        optimizer_name: str = "COBYLA",
        max_iter: int = 50,
        random_seed: int = 42,
        shots: int = 1024
    ):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.optimizer_name = optimizer_name.upper()
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.shots = shots
        self.backend_name = "Local Quantum Simulator (Qiskit Aer SamplerV2)"
        self.encoding_type = "Angle Encoding [Ry(theta)]"
        self.classifier_type = "Variational Quantum Classifier (VQC)"
        self.is_trained = False
        
        # Build circuits
        self.feature_map = build_angle_encoding_circuit(num_qubits=num_qubits)
        self.ansatz = build_variational_ansatz(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
        
        # Use StatevectorSampler from core Qiskit for reliable, cross-platform simulation
        self.sampler = StatevectorSampler(seed=random_seed, default_shots=shots)
        self.backend_name = "Local Quantum Simulator (StatevectorSampler)"
            
        # Select optimizer
        if self.optimizer_name == "SPSA":
            self.optimizer = SPSA(maxiter=max_iter)
        else:
            self.optimizer = COBYLA(maxiter=max_iter)
        
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            sampler=self.sampler
        )
        self.weights = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the VQC using classical optimization over quantum circuit parameters.
        """
        print(f"[Quantum VQC] Initiating training on {len(X_train)} samples across {self.num_qubits} qubits...")
        print(f"[Quantum VQC] Optimizer: COBYLA (maxiter={self.max_iter}), Backend: {self.backend_name}")
        
        # Train VQC
        self.vqc.fit(X_train, y_train)
        self.weights = np.array(self.vqc.weights)
        self.is_trained = True
        print(f"[Quantum VQC] Training complete! Optimized {len(self.weights)} variational parameters.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary class {0: Legitimate, 1: Fraudulent}."""
        if not self.is_trained:
            raise RuntimeError("Quantum model has not been trained yet.")
        return self.vqc.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability array of shape (N, 2): [P(legit), P(fraud)].
        """
        if not self.is_trained:
            raise RuntimeError("Quantum model has not been trained yet.")
        try:
            probs = self.vqc.predict_proba(X)
        except Exception as e:
            print(f"[Quantum VQC] Sampler error ({e}), re-evaluating with pure StatevectorSampler...")
            self.sampler = StatevectorSampler()
            self.vqc = VQC(
                feature_map=self.feature_map,
                ansatz=self.ansatz,
                optimizer=self.optimizer,
                sampler=self.sampler
            )
            from types import SimpleNamespace
            self.vqc._fit_result = SimpleNamespace(x=self.weights, fun=0.0)
            probs = self.vqc.predict_proba(X)

        # Ensure proper normalization
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def get_fraud_risk_score(self, x_single: np.ndarray) -> float:
        """
        Returns fraud risk score as a percentage [0.0% to 100.0%]
        for a single transaction feature vector.
        """
        if x_single.ndim == 1:
            x_single = x_single.reshape(1, -1)
        proba = self.predict_proba(x_single)
        # Class 1 is Fraud
        fraud_risk = float(proba[0, 1]) * 100.0
        return round(fraud_risk, 2)

    def get_full_circuit(self):
        """Returns the complete combined QuantumCircuit for visualization."""
        return build_full_quantum_circuit(
            num_qubits=self.num_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )

    def save(self, model_dir: str = "models"):
        """
        Saves the VQC model using modern to_dill() and saves weights array + metadata.
        """
        os.makedirs(model_dir, exist_ok=True)
        dill_path = os.path.join(model_dir, "quantum_vqc.dill")
        self.vqc.to_dill(dill_path)
        
        # Also save weights & config as transparent standalone backup
        weights_path = os.path.join(model_dir, "vqc_weights.npy")
        np.save(weights_path, self.weights)
        
        meta = {
            "num_qubits": self.num_qubits,
            "reps": self.reps,
            "entanglement": self.entanglement,
            "optimizer_name": self.optimizer_name,
            "max_iter": self.max_iter,
            "shots": self.shots,
            "backend_name": self.backend_name,
            "encoding_type": self.encoding_type,
            "classifier_type": self.classifier_type,
            "num_weights": len(self.weights) if self.weights is not None else 0
        }
        meta_path = os.path.join(model_dir, "quantum_model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        print(f"[Quantum VQC] Saved model artifacts to '{model_dir}/'.")

    @classmethod
    def load(cls, model_dir: str = "models"):
        """Loads trained QuantumFraudVQC from disk."""
        dill_path = os.path.join(model_dir, "quantum_vqc.dill")
        meta_path = os.path.join(model_dir, "quantum_model_meta.json")
        weights_path = os.path.join(model_dir, "vqc_weights.npy")
        
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
        model_obj = cls(
            num_qubits=meta.get("num_qubits", 4),
            reps=meta.get("reps", 1),
            entanglement=meta.get("entanglement", "linear"),
            optimizer_name=meta.get("optimizer_name", "COBYLA"),
            max_iter=meta.get("max_iter", 50),
            shots=meta.get("shots", 1024)
        )
        
        loaded = False
        # 1. Primary: Restore from direct weights array using pure StatevectorSampler (zero C++ dependencies)
        if os.path.exists(weights_path):
            try:
                from types import SimpleNamespace
                weights = np.load(weights_path)
                model_obj.weights = weights
                model_obj.vqc._fit_result = SimpleNamespace(x=weights, fun=0.0)
                model_obj.is_trained = True
                loaded = True
                print(f"[Quantum VQC] Successfully restored VQC with StatevectorSampler from '{weights_path}'.")
            except Exception as e:
                print(f"[Quantum VQC] Direct weights restoration failed: {e}")

        # 2. Secondary fallback: Attempt dill
        if not loaded and os.path.exists(dill_path):
            try:
                model_obj.vqc = VQC.from_dill(dill_path)
                model_obj.weights = np.array(model_obj.vqc.weights)
                model_obj.is_trained = True
                loaded = True
                print(f"[Quantum VQC] Successfully loaded VQC model from '{dill_path}'.")
            except Exception as e:
                print(f"[Quantum VQC] dill deserialization failed ({e})")

        if not loaded:
            raise FileNotFoundError(f"Could not load VQC from weights or dill in {model_dir}")
            
        return model_obj

if __name__ == "__main__":
    # Smoke test on small random dataset
    q_model = QuantumFraudVQC(num_qubits=4, reps=1, max_iter=10)
    X_toy = np.random.uniform(0, np.pi, size=(10, 4))
    y_toy = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    q_model.fit(X_toy, y_toy)
    pred = q_model.predict(X_toy[:2])
    score = q_model.get_fraud_risk_score(X_toy[0])
    print(f"Predictions: {pred}, Risk Score: {score}%")
    q_model.save("models_test")
    loaded = QuantumFraudVQC.load("models_test")
    print("Loaded score:", loaded.get_fraud_risk_score(X_toy[0]))
    # Clean up test dir
    import shutil
    shutil.rmtree("models_test")
    print("Quantum VQC module verified successfully!")
