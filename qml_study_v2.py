import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

# Qiskit imports
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute

# Fixed seed for reproducibility
SEED = 42
algorithm_globals.random_seed = SEED

class DefectPipeline:
    def __init__(self, dataset_path, target_col='bugs'):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.dataset_name = os.path.basename(dataset_path).replace('.csv', '')
        self.results_dir = os.path.join("results_plots", "v2", self.dataset_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_and_explore(self):
        print(f"\n--- Loading Dataset: {self.dataset_name} ---")
        df = pd.read_csv(self.dataset_path, sep=';')
        df.columns = [c.strip() for c in df.columns]
        
        # Binary target: bugs > 0
        y = (df[self.target_col] > 0).astype(int)
        
        # Feature columns: drop non-features
        cols_to_drop = ['classname', 'bugs', 'nonTrivialBugs', 'majorBugs', 'criticalBugs', 'highPriorityBugs', '']
        X = df.drop(cols_to_drop, axis=1, errors='ignore')
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        print(f"Dataset Shape: {X.shape}")
        print(f"Class Distribution:\n{y.value_counts()}")
        return X, y

    def preprocess(self, X, y, apply_smote=True, n_features_quantum=2):
        # 1. Stratified Split (on original data to avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED
        )
        
        # 2. Scaling (Fit on Train Only)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # 3. SMOTE (Applied to Training Data ONLY)
        X_train_final, y_train_final = X_train_std, y_train
        if apply_smote:
            print(f"Applying SMOTE...")
            smote = SMOTE(random_state=SEED)
            X_train_final, y_train_final = smote.fit_resample(X_train_std, y_train)
            print(f"New Class Distribution (Train): {pd.Series(y_train_final).value_counts().to_dict()}")
        
        # 4. Feature Selection for Quantum (SelectKBest)
        selector = SelectKBest(f_classif, k=n_features_quantum)
        X_train_q = selector.fit_transform(X_train_final, y_train_final)
        X_test_q = selector.transform(X_test_std)
        
        # Map to [0, pi] for Angle Encoding
        q_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_q = q_scaler.fit_transform(X_train_q)
        X_test_q = q_scaler.transform(X_test_q)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected Features for Quantum: {selected_features}")
        
        return (X_train_final, X_test_std, y_train_final, y_test), (X_train_q, X_test_q, y_train_final, y_test)

    def get_noise_model(self):
        noise_model = NoiseModel()
        p1q = 0.02
        p2q = 0.08
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), ['rx', 'ry', 'rz', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ['cx', 'cz'])
        return noise_model

    def evaluate(self, model_name, config, y_true, y_pred, train_time):
        metrics = {
            "Model": model_name,
            "Config": config,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0, pos_label=1),
            "Recall": recall_score(y_true, y_pred, zero_division=0, pos_label=1),
            "F1": f1_score(y_true, y_pred, zero_division=0, pos_label=1),
            "TrainTime": round(train_time, 3),
            "DatasetName": self.dataset_name
        }
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} ({config})\nCM: {self.dataset_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{model_name}_{config}_cm.png"))
        plt.close()
        
        return metrics

def run_experiment(dataset_path):
    pipe = DefectPipeline(dataset_path)
    X, y = pipe.load_and_explore()
    
    # We will test two configurations: Original vs SMOTE
    all_metrics = []
    
    for smote_toggle in [False, True]:
        config_suffix = "SMOTE" if smote_toggle else "Original"
        print(f"\n>>> Running Config: {config_suffix} <<<")
        
        # For Quantum simulation speed, we subset the preprocessed data to 100 samples
        # but only AFTER splitting/SMOTEing to maintain scientific validity.
        # However, for a fair comparison, we use the full training set for SMOTE/Selection.
        
        (X_tr_c, X_te_c, y_tr_c, y_te_c), (X_tr_q, X_te_q, y_tr_q, y_te_q) = pipe.preprocess(X, y, apply_smote=smote_toggle)
        
        # Subsampling for Quantum (due to time constraints)
        LIMIT_TRAIN = 100
        LIMIT_TEST = 50 # Significantly speeds up QSVC and VQC predictions

        train_indices = np.random.choice(len(X_tr_q), min(LIMIT_TRAIN, len(X_tr_q)), replace=False)
        X_tr_q_sub = X_tr_q[train_indices]
        y_tr_q_sub = y_tr_q.iloc[train_indices] if isinstance(y_tr_q, pd.Series) else y_tr_q[train_indices]

        test_indices = np.random.choice(len(X_te_q), min(LIMIT_TEST, len(X_te_q)), replace=False)
        X_te_q_sub = X_te_q[test_indices]
        y_te_q_sub = y_te_q.iloc[test_indices] if isinstance(y_te_q, pd.Series) else y_te_q[test_indices]
        
        # 1. Classical Baseline
        print("- Training Classical SVM...")
        start = time.time()
        svm = SVC(kernel='rbf', random_state=SEED)
        svm.fit(X_tr_c, y_tr_c)
        y_pred = svm.predict(X_te_c)
        all_metrics.append(pipe.evaluate("Classical SVM", config_suffix, y_te_c, y_pred, time.time() - start))

        # Setup Quantum Engines
        n_qubits = X_tr_q.shape[1]
        feature_map = ZZFeatureMap(n_qubits, reps=2)
        ansatz = RealAmplitudes(n_qubits, reps=2)
        optimizer = COBYLA(maxiter=50) # Reduced from 100 for speed
        
        # Optimize Aer for small qubit counts (statevector is fastest)
        backend = AerSimulator(method='statevector')
        ideal_sampler = AerSamplerV2()
        noise_model = pipe.get_noise_model()
        noisy_sampler = AerSamplerV2(options={"backend_options": {"noise_model": noise_model, "method": "statevector"}})
        pm = generate_preset_pass_manager(1, backend)

        # 2. VQC Ideal
        print("- Training VQC Ideal...")
        start = time.time()
        vqc_ideal = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=ideal_sampler, pass_manager=pm)
        vqc_ideal.fit(X_tr_q_sub, np.eye(2)[y_tr_q_sub.astype(int)])
        y_pred = np.argmax(vqc_ideal.predict(X_te_q_sub), axis=1)
        all_metrics.append(pipe.evaluate("VQC", f"{config_suffix}_Ideal", y_te_q_sub, y_pred, time.time() - start))

        # 3. VQC Noisy
        print("- Training VQC Noisy...")
        start = time.time()
        vqc_noisy = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=noisy_sampler, pass_manager=pm)
        vqc_noisy.fit(X_tr_q_sub, np.eye(2)[y_tr_q_sub.astype(int)])
        y_pred = np.argmax(vqc_noisy.predict(X_te_q_sub), axis=1)
        all_metrics.append(pipe.evaluate("VQC", f"{config_suffix}_Noisy", y_te_q_sub, y_pred, time.time() - start))

        # 4. QSVC Ideal
        print("- Training QSVC Ideal...")
        start = time.time()
        # V2 samplers need a transpiler in ComputeUncompute
        fidelity_ideal = ComputeUncompute(sampler=ideal_sampler, transpiler=pm)
        kernel_ideal = FidelityQuantumKernel(fidelity=fidelity_ideal)
        qsvc_ideal = QSVC(quantum_kernel=kernel_ideal)
        qsvc_ideal.fit(X_tr_q_sub, y_tr_q_sub)
        y_pred = qsvc_ideal.predict(X_te_q_sub)
        all_metrics.append(pipe.evaluate("QSVC", f"{config_suffix}_Ideal", y_te_q_sub, y_pred, time.time() - start))

    # Save metrics to CSV
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(pipe.results_dir, "dataset_metrics.csv"), index=False)
    
    # Summary Bar Plots
    plt.figure(figsize=(12, 6))
    melted = df_metrics.melt(id_vars=['Model', 'Config'], value_vars=['Balanced Acc', 'F1'])
    sns.barplot(data=melted, x='Model', y='value', hue='Config')
    plt.title(f"Performance Comparison: {pipe.dataset_name}")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(pipe.results_dir, "summary_performance.png"))
    plt.close()
    
    return all_metrics

if __name__ == "__main__":
    dataset_dir = "dataset"
    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    dataset_files.sort()
    
    master_results = []
    
    print(f"Starting Multi-Dataset Research Suite v2...")
    print(f"Target Datasets: {dataset_files}")
    
    for dataset_file in dataset_files:
        dataset_path = os.path.join(dataset_dir, dataset_file)
        results = run_experiment(dataset_path)
        master_results.extend(results)
    
    # Save master results to CSV
    master_df = pd.DataFrame(master_results)
    os.makedirs("results_plots/v2", exist_ok=True)
    master_df.to_csv("results_plots/v2/combined_performance_metrics_v2.csv", index=False)
    
    print("\n====================================================================================")
    print("Full Multi-Dataset Benchmark Complete!")
    print(f"Results consolidated in: results_plots/v2/combined_performance_metrics_v2.csv")
    print("====================================================================================")
