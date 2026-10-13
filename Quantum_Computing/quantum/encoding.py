"""
quantum/encoding.py
-------------------
Quantum Feature Encoding for 4-qubit Financial Fraud Detection.
Implements Angle Encoding mapping normalized continuous features into
single-qubit rotation gates: Ry(theta).
"""

from qiskit.circuit import QuantumCircuit, ParameterVector

def build_angle_encoding_circuit(num_qubits: int = 4, param_name: str = "x") -> QuantumCircuit:
    """
    Constructs an Angle Encoding feature map on `num_qubits` qubits.
    Initial State: |0000>
    Gate applied to each qubit i: Ry(x[i])
    where x[i] in [0, pi] is the scaled feature value:
      q0: transaction_amount       -> Ry(theta0)
      q1: transaction_frequency    -> Ry(theta1)
      q2: location_risk            -> Ry(theta2)
      q3: device_changed           -> Ry(theta3)

    Returns:
    --------
    qiskit.circuit.QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits, name="AngleEncoding")
    params = ParameterVector(param_name, num_qubits)
    
    for i in range(num_qubits):
        qc.ry(params[i], i)
        
    return qc

if __name__ == "__main__":
    qc = build_angle_encoding_circuit(4)
    print("Angle Encoding Circuit:")
    print(qc)
