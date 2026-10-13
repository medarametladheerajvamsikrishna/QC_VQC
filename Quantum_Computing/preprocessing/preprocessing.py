"""
preprocessing/preprocessing.py
------------------------------
Data preprocessing pipeline for the Quantum AI Fraud Detection prototype.
Handles cleaning, stratified train/test split, scaling without leakage,
and feature subset extraction for the 4-qubit Quantum circuit.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Selected top 4 informative features for the 4-qubit quantum classifier
QUANTUM_FEATURES = [
    "failed_attempts",
    "location_risk",
    "account_age_days",
    "merchant_risk"
]

ALL_FEATURES = [
    "transaction_amount",
    "transaction_frequency",
    "location_risk",
    "device_changed",
    "previous_fraud",
    "distance_from_usual_location",
    "transaction_hour",
    "merchant_risk",
    "failed_attempts",
    "international_transaction",
    "account_age_days"
]

TARGET_COL = "fraud_label"

def load_and_clean_data(filepath: str = "data/transactions.csv") -> pd.DataFrame:
    """
    Loads dataset, checks for nulls and duplicates, cleans if necessary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run generate_dataset.py first.")
    
    df = pd.read_csv(filepath)
    initial_len = len(df)
    
    # Check & handle missing values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        df = df.dropna()
        print(f"[Preprocessing] Dropped {null_counts} missing value rows.")
        
    # Check & handle duplicates
    dup_counts = df.duplicated().sum()
    if dup_counts > 0:
        df = df.drop_duplicates()
        print(f"[Preprocessing] Removed {dup_counts} duplicate records.")
        
    print(f"[Preprocessing] Loaded {len(df)} transactions from {filepath} (Cleaned {initial_len - len(df)} records).")
    return df

def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_dir: str = "models"
):
    """
    Splits data into train/test sets with stratification.
    Fits scalers ONLY on training data to prevent data leakage.
    Returns:
        X_train_full, X_test_full,
        X_train_q_scaled, X_test_q_scaled,
        y_train, y_test,
        full_scaler, quantum_scaler
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Full feature scaler for classical baseline (StandardScaler)
    full_scaler = StandardScaler()
    X_train_full_scaled = full_scaler.fit_transform(X_train)
    X_test_full_scaled = full_scaler.transform(X_test)

    # Quantum feature subset (4 features)
    X_train_q = X_train[QUANTUM_FEATURES].copy()
    X_test_q = X_test[QUANTUM_FEATURES].copy()

    # Quantum scaler: MinMax scaled to [0, np.pi] for Angle Encoding Ry(theta)
    quantum_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    X_train_q_scaled = quantum_scaler.fit_transform(X_train_q)
    X_test_q_scaled = quantum_scaler.transform(X_test_q)

    # Save scalers for live inference
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(full_scaler, os.path.join(scaler_dir, "full_scaler.joblib"))
    joblib.dump(quantum_scaler, os.path.join(scaler_dir, "quantum_scaler.joblib"))
    print(f"[Preprocessing] Saved scalers to '{scaler_dir}/'.")

    return {
        "X_train_full": X_train_full_scaled,
        "X_test_full": X_test_full_scaled,
        "X_train_q": X_train_q_scaled,
        "X_test_q": X_test_q_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "full_scaler": full_scaler,
        "quantum_scaler": quantum_scaler
    }

def get_quantum_training_subset(
    X_q: np.ndarray,
    y: np.ndarray,
    n_samples: int = 600,
    fraud_weight: float = 0.35,
    random_state: int = 42
):
    """
    Extracts a balanced/representative subset for laptop-friendly VQC training.
    Fraud datasets are highly imbalanced (~7% fraud), which in small random batches
    causes gradient starvation. Here we stratify with configurable fraud weighting
    (e.g., 35% fraud in training subset) so the VQC learns the decision boundary efficiently,
    while evaluating strictly on the full natural test distribution!
    """
    rng = np.random.default_rng(random_state)
    
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    
    n_fraud = int(n_samples * fraud_weight)
    n_legit = n_samples - n_fraud
    
    # Cap if needed
    n_fraud = min(n_fraud, len(fraud_idx))
    n_legit = min(n_legit, len(legit_idx))
    
    selected_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
    selected_legit = rng.choice(legit_idx, size=n_legit, replace=False)
    
    subset_idx = np.concatenate([selected_fraud, selected_legit])
    rng.shuffle(subset_idx)
    
    print(f"[Quantum Subset] Selected {len(subset_idx)} samples ({n_legit} legit, {n_fraud} fraud) for VQC training.")
    return X_q[subset_idx], y[subset_idx]

if __name__ == "__main__":
    df = load_and_clean_data()
    data = prepare_data(df)
    print(f"X_train shape: {data['X_train_full'].shape}")
    print(f"X_test shape: {data['X_test_full'].shape}")
    print(f"X_train_q shape: {data['X_train_q'].shape}")
    X_sub, y_sub = get_quantum_training_subset(data['X_train_q'], data['y_train'], n_samples=500)
    print(f"Subset shapes: X={X_sub.shape}, y={y_sub.shape}")
