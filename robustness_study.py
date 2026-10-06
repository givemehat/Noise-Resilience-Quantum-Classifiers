import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import warnings

warnings.filterwarnings('ignore')

SEEDS = [42, 100, 2023, 777, 1234]

NOISE_LEVELS = {
    'Ideal': (0.0, 0.0),
    'Low_Stress': (0.01, 0.04),
    'High_Stress': (0.02, 0.08)
}

def get_datasets():
    datasets = {}
    
    # 1. Binary Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    idx = y < 2
    datasets['Binary_Iris'] = (X[idx], y[idx])
    
    # 2. Make Moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    datasets['Make_Moons'] = (X, y)
    
    # 3. Make Circles
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    datasets['Make_Circles'] = (X, y)
    
    # 4. Breast Cancer (downsampled for simulation constraints)
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X, _, y, _ = train_test_split(X, y, train_size=100, stratify=y, random_state=42)
    datasets['Breast_Cancer'] = (X, y)
    
    # 5. Wine (Binary subset, downsampled)
    wine = load_wine()
    X, y = wine.data, wine.target
    idx = y < 2
    X_wine, y_wine = X[idx], y[idx]
    X_wine, _, y_wine, _ = train_test_split(X_wine, y_wine, train_size=100, stratify=y_wine, random_state=42)
    datasets['Wine'] = (X_wine, y_wine)
    
    return datasets

def preprocess_data(X, y, random_state, n_features=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
    
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)
    
    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = angle_scaler.fit_transform(X_train_pca)
    X_test_final = angle_scaler.transform(X_test_pca)
    
    return X_train_final, X_test_final, y_train, y_test

def extract_circuit_metrics(circuit, pm):
    transpiled = pm.run(circuit)
    ops = transpiled.count_ops()
    cnot_count = ops.get('cx', 0)
    gates_1q = sum(c for g, c in ops.items() if g not in ['cx', 'barrier', 'measure'])
    return transpiled.depth(), gates_1q, cnot_count

def run_experiments():
    datasets = get_datasets()
    results = []
    
    pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    
    # Determine gates for the noise model dynamically based on circuit transpilation
    dummy_feature_map = ZZFeatureMap(2, reps=2)
    dummy_ansatz = RealAmplitudes(2, reps=2)
    t_vqc = pm.run(dummy_feature_map.compose(dummy_ansatz))
    t_kernel = pm.run(dummy_feature_map.compose(dummy_feature_map.inverse()))
    
    all_ops = set(t_vqc.count_ops().keys()).union(set(t_kernel.count_ops().keys()))
    all_ops.discard('barrier')
    all_ops.discard('measure')
    gate_1q = [g for g in all_ops if g != 'cx']
    gate_2q = ['cx']
    
    for ds_name, (X, y) in datasets.items():
        for seed in SEEDS:
            # Seed everything for strict reproducibility
            np.random.seed(seed)
            algorithm_globals.random_seed = seed
            
            X_train, X_test, y_train, y_test = preprocess_data(X, y, random_state=seed)
            n_qubits = X_train.shape[1]
            
            feature_map = ZZFeatureMap(n_qubits, reps=2)
            ansatz = RealAmplitudes(n_qubits, reps=2)
            
            for noise_name, (p1, p2) in NOISE_LEVELS.items():
                if noise_name == 'Ideal':
                    sampler = StatevectorSampler()
                else:
                    noise_model = NoiseModel()
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), gate_1q)
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), gate_2q)
                    sampler = AerSamplerV2(default_shots=1024, options={'backend_options': {'noise_model': noise_model, 'seed_simulator': seed}})
                
                # Classical RBF-SVM Baseline (evaluated once per seed)
                if noise_name == 'Ideal':
                    svm = SVC(kernel='rbf', random_state=seed)
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)
                    results.append({
                        'Dataset': ds_name, 'Seed': seed, 'Noise': 'Ideal', 'Model': 'Classical_SVM',
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, zero_division=0),
                        'Recall': recall_score(y_test, y_pred, zero_division=0),
                        'F1': f1_score(y_test, y_pred, zero_division=0)
                    })
                
                # Quantum Support Vector Classifier
                np.random.seed(seed)
                algorithm_globals.random_seed = seed
                fidelity = ComputeUncompute(sampler=sampler, transpiler=pm)
                qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity))
                qsvc.fit(X_train, y_train)
                y_pred_qsvc = qsvc.predict(X_test)
                results.append({
                    'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': 'QSVC',
                    'Accuracy': accuracy_score(y_test, y_pred_qsvc),
                    'Precision': precision_score(y_test, y_pred_qsvc, zero_division=0),
                    'Recall': recall_score(y_test, y_pred_qsvc, zero_division=0),
                    'F1': f1_score(y_test, y_pred_qsvc, zero_division=0)
                })
                
                # Variational Quantum Classifiers (COBYLA & SPSA)
                for opt_name, optimizer in [('VQC_COBYLA_300', COBYLA(maxiter=300)), ('VQC_SPSA_100', SPSA(maxiter=100))]:
                    np.random.seed(seed)
                    algorithm_globals.random_seed = seed
                    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=sampler, pass_manager=pm)
                    vqc.fit(X_train, np.eye(2)[y_train])
                    y_pred_vqc = np.argmax(vqc.predict(X_test), axis=1)
                    results.append({
                        'Dataset': ds_name, 'Seed': seed, 'Noise': noise_name, 'Model': opt_name,
                        'Accuracy': accuracy_score(y_test, y_pred_vqc),
                        'Precision': precision_score(y_test, y_pred_vqc, zero_division=0),
                        'Recall': recall_score(y_test, y_pred_vqc, zero_division=0),
                        'F1': f1_score(y_test, y_pred_vqc, zero_division=0)
                    })

if __name__ == '__main__':
    run_experiments()
