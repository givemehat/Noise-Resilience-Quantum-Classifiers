import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import os
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
    idx = y < 2 # Binary classification
    datasets['Binary_Iris'] = (X[idx], y[idx])
    
    # 2. Make Moons
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    datasets['Make_Moons'] = (X, y)
    
    # 3. Breast Cancer (downsampled)
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    X, _, y, _ = train_test_split(X, y, train_size=100, stratify=y, random_state=42)
    datasets['Breast_Cancer'] = (X, y)
    
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
    transpiled_circuit = pm.run(circuit)
    depth = transpiled_circuit.depth()
    ops = transpiled_circuit.count_ops()
    cnot_count = ops.get('cx', 0)
    gates_1q = sum(count for gate, count in ops.items() if gate not in ['cx', 'barrier', 'measure'])
    return depth, gates_1q, cnot_count

def run_experiments():
    results = []
    circuit_metrics = []
    datasets = get_datasets()
    print("Starting QML Robustness Experiments...")
    
    # For VQC we will evaluate both COBYLA (increased budget) and SPSA
    optimizers_to_test = {
        'VQC_COBYLA': COBYLA(maxiter=300),
        'VQC_SPSA': SPSA(maxiter=100) # SPSA takes more shots per iter, 100 is equivalent robust budget
    }
    
    for ds_name, (X, y) in datasets.items():
        print(f"\n[ Dataset: {ds_name} ]")
        
        for seed in SEEDS:
            print(f"  > Seed: {seed}")
            X_train, X_test, y_train, y_test = preprocess_data(X, y, random_state=seed, n_features=2)
            
            # Classical baseline
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
            
            n_qubits = X_train.shape[1]
            feature_map = ZZFeatureMap(n_qubits, reps=2)
            ansatz = RealAmplitudes(n_qubits, reps=2)
            
            for noise_name, (p1, p2) in NOISE_LEVELS.items():
                if p1 == 0 and p2 == 0:
                    sampler = AerSamplerV2(default_shots=1024)
                else:
                    noise_model = NoiseModel()
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx', 'ry', 'rz', 'h', 'x'])
                    noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx', 'cz'])
                    sampler = AerSamplerV2(default_shots=1024, options={'backend_options': {'noise_model': noise_model, 'method': 'statevector'}})
                
                pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
                
                # Circuit Metrics
                if seed == SEEDS[0] and noise_name == 'Ideal':
                    v_depth, v_1q, v_cx = extract_circuit_metrics(feature_map.compose(ansatz), pm)
                    circuit_metrics.append({'Dataset': ds_name, 'Model': 'VQC_Circuit', 'Depth': v_depth, '1Q_Gates': v_1q, 'CNOTs': v_cx})
                    k_depth, k_1q, k_cx = extract_circuit_metrics(feature_map.compose(feature_map.inverse()), pm)
                    circuit_metrics.append({'Dataset': ds_name, 'Model': 'ComputeUncompute_Kernel', 'Depth': k_depth, '1Q_Gates': k_1q, 'CNOTs': k_cx})

                # Train VQCs (COBYLA and SPSA)
                for opt_name, optimizer in optimizers_to_test.items():
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
                
                # QSVC
                qsvc = QSVC(quantum_kernel=FidelityQuantumKernel(fidelity=ComputeUncompute(sampler=sampler, transpiler=pm)))
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
    df_metrics = pd.DataFrame(circuit_metrics).drop_duplicates()
    
    os.makedirs('results_revision', exist_ok=True)
    df_res.to_csv('results_revision/raw_results.csv', index=False)
    df_metrics.to_csv('results_revision/circuit_metrics.csv', index=False)
    
    generate_plots(df_res)
    print("\nExperiments complete. Results and plots saved to 'results_revision' directory.")

def generate_plots(df):
    os.makedirs('results_revision/plots', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    for ds in df['Dataset'].unique():
        df_ds = df[df['Dataset'] == ds]
        
        plt.figure(figsize=(10, 6))
        # Plot with error bars (seaborn barplot automatically calculates mean and 95% CI or std if estimator is set)
        sns.barplot(data=df_ds, x='Noise', y='F1', hue='Model', capsize=.1, errorbar='sd')
        plt.title(f'F1 Score across Noise Levels: {ds}\n(Mean ± Std across 5 seeds)', fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results_revision/plots/{ds}_F1_Scores.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_ds, x='Noise', y='Accuracy', hue='Model', capsize=.1, errorbar='sd')
        plt.title(f'Accuracy across Noise Levels: {ds}\n(Mean ± Std across 5 seeds)', fontsize=14)
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results_revision/plots/{ds}_Accuracy.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    run_experiments()
