import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time
import os
import warnings

warnings.filterwarnings('ignore')

SEEDS = [42, 100, 2023, 777, 1234]

# Defining 3 noise levels: None (ideal), Low (p1=0.01, p2=0.04), High (p1=0.02, p2=0.08)
NOISE_LEVELS = {
    'Ideal': (0.0, 0.0),
    'Low_Stress': (0.01, 0.04),
    'High_Stress': (0.02, 0.08)
}

def get_datasets():
    datasets = {}
    
    # 1. Binary Iris (for continuity)
    iris = load_iris()
    X, y = iris.data, iris.target
    idx = y < 2 # Binary classification
    datasets['Binary_Iris'] = (X[idx], y[idx])
    
    # 2. Make Moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    datasets['Make_Moons'] = (X, y)
    
    # 3. Make Circles
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    datasets['Make_Circles'] = (X, y)
    
    # 4. Breast Cancer (downsampled)
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X, _, y, _ = train_test_split(X, y, train_size=100, stratify=y, random_state=42)
    datasets['Breast_Cancer'] = (X, y)
    
    return datasets

def preprocess_data(X, y, random_state, n_features=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    
    # Standard scale
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    # PCA to reduce to n_features for quantum simulation efficiency
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    # Angle Encoding [0, pi]
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = angle_scaler.fit_transform(X_train_pca)
    X_test_final = angle_scaler.transform(X_test_pca)
    
    return X_train_final, X_test_final, y_train, y_test

def extract_circuit_metrics(circuit, pm):
    # Transpile the circuit
    transpiled_circuit = pm.run(circuit)
    
    # Extract depth and gate counts
    depth = transpiled_circuit.depth()
    ops = transpiled_circuit.count_ops()
    cnot_count = ops.get('cx', 0)
    
    # All 1-qubit gates
    gates_1q = sum(count for gate, count in ops.items() if gate != 'cx' and gate != 'barrier' and gate != 'measure')
    
    return depth, gates_1q, cnot_count

def run_experiments():
    results = []
    circuit_metrics = []
    
    datasets = get_datasets()
    
    print("Starting QML Revision Experiments...")
    
    for ds_name, (X, y) in datasets.items():
        print(f"\nEvaluating Dataset: {ds_name}")
        
        for seed in SEEDS:
            print(f"  Seed: {seed}")
            X_train, X_test, y_train, y_test = preprocess_data(X, y, random_state=seed, n_features=2)
            
            # Classical baseline
            svm = SVC(kernel='rbf', random_state=seed)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            results.append({
                'Dataset': ds_name, 'Seed': seed, 'Noise': 'None', 'Model': 'Classical_SVM',
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0)
            })
            
            # Quantum setups
            n_qubits = X_train.shape[1]
            feature_map = ZZFeatureMap(n_qubits, reps=2)
            ansatz = RealAmplitudes(n_qubits, reps=2)
            # Increased budget or using SPSA
            optimizer = SPSA(maxiter=100) # SPSA is better for noise and equivalent to ~300 COBYLA evals in robustness
            
            for noise_name, (p1, p2) in NOISE_LEVELS.items():
                
                # Setup noise and sampler
                if p1 == 0 and p2 == 0:
                    sampler = AerSamplerV2(default_shots=1024)
                else:
                    noise_model = NoiseModel()
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx', 'ry', 'rz', 'h', 'x'])
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx', 'cz'])
                    sampler = AerSamplerV2(default_shots=1024, options={'backend_options': {'noise_model': noise_model}})
                
                pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
                
                # Extract Circuit Metrics only once per dataset (independent of noise/seed)
                if seed == SEEDS[0] and noise_name == 'Ideal':
                    # VQC circuit
                    vqc_circuit = feature_map.compose(ansatz)
                    v_depth, v_1q, v_cx = extract_circuit_metrics(vqc_circuit, pm)
                    circuit_metrics.append({'Dataset': ds_name, 'Model': 'VQC_Circuit', 'Depth': v_depth, '1Q_Gates': v_1q, 'CNOTs': v_cx})
                    
                    # Compute-Uncompute Kernel Circuit
                    # Compute-uncompute uses feature map + adjoint
                    kernel_circuit = feature_map.compose(feature_map.inverse())
                    k_depth, k_1q, k_cx = extract_circuit_metrics(kernel_circuit, pm)
                    circuit_metrics.append({'Dataset': ds_name, 'Model': 'ComputeUncompute_Kernel', 'Depth': k_depth, '1Q_Gates': k_1q, 'CNOTs': k_cx})

                # VQC
                vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler, pass_manager=pm)
                y_train_vqc = np.eye(2)[y_train]
                vqc.fit(X_train, y_train_vqc)
                y_pred_vqc = np.argmax(vqc.predict(X_test), axis=1)
                
                results.append({
                    'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': 'VQC',
                    'Accuracy': accuracy_score(y_test, y_pred_vqc),
                    'Precision': precision_score(y_test, y_pred_vqc, zero_division=0),
                    'Recall': recall_score(y_test, y_pred_vqc, zero_division=0),
                    'F1': f1_score(y_test, y_pred_vqc, zero_division=0)
                })
                
                # QSVC
                fidelity = ComputeUncompute(sampler=sampler, transpiler=pm)
                kernel = FidelityQuantumKernel(fidelity=fidelity)
                qsvc = QSVC(quantum_kernel=kernel)
                qsvc.fit(X_train, y_train)
                y_pred_qsvc = qsvc.predict(X_test)
                
                results.append({
                    'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': 'QSVC',
                    'Accuracy': accuracy_score(y_test, y_pred_qsvc),
                    'Precision': precision_score(y_test, y_pred_qsvc, zero_division=0),
                    'Recall': recall_score(y_test, y_pred_qsvc, zero_division=0),
                    'F1': f1_score(y_test, y_pred_qsvc, zero_division=0)
                })

    df_res = pd.DataFrame(results)
    df_metrics = pd.DataFrame(circuit_metrics)
    
    # Calculate Mean and Std
    df_agg = df_res.groupby(['Dataset', 'Noise', 'Model']).agg({
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1': ['mean', 'std']
    }).reset_index()
    
    os.makedirs('results_revision', exist_ok=True)
    df_res.to_csv('results_revision/raw_results.csv', index=False)
    df_agg.to_csv('results_revision/aggregated_results.csv', index=False)
    df_metrics.to_csv('results_revision/circuit_metrics.csv', index=False)
    
    print("\nExperiments complete. Results saved to 'results_revision' folder.")
    print("\n--- Circuit Metrics ---")
    print(df_metrics)

if __name__ == '__main__':
    run_experiments()
