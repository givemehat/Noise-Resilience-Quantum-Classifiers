import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes  # type: ignore
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore
from qiskit_algorithms.optimizers import COBYLA  # type: ignore
from qiskit_algorithms.state_fidelities import ComputeUncompute  # type: ignore
from qiskit_algorithms.utils import algorithm_globals  # type: ignore

# Modern Qiskit Machine Learning imports (no deprecated `sampler=` in kernel constructor)
from qiskit_machine_learning.algorithms import VQC, QSVC  # type: ignore
from qiskit_machine_learning.kernels import FidelityQuantumKernel  # type: ignore

# Modern Aer imports
from qiskit_aer import AerSimulator  # type: ignore
from qiskit_aer.noise import NoiseModel, depolarizing_error  # type: ignore
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2  # type: ignore


def load_and_preprocess_data(file_path):
    """Load a CSV dataset, clean columns, set binary labels, sample 100 rows, and preprocess."""
    df = pd.read_csv(file_path, sep=';')
    
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Define binary target: Class 1 if bugs > 0, else Class 0
    target_col = 'bugs'
    if target_col not in df.columns:
        # Fallback if bugs col is missing
        target_col = df.columns[-2] 

    y_full = (df[target_col] > 0).astype(int)
    
    # Drop non-feature and other target columns
    cols_to_drop = ['classname', 'bugs', 'nonTrivialBugs', 'majorBugs', 'criticalBugs', 'highPriorityBugs', '']
    X_full = df.drop(cols_to_drop, axis=1, errors='ignore')
    
    # Ensure all data is numeric for scaling
    X_full = X_full.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Sample 100 rows to keep simulation time manageable
    sample_size = min(100, len(df))
    from sklearn.model_selection import train_test_split as sampler_split
    
    # Ensure we have at least 2 classes for stratification
    if len(np.unique(y_full)) > 1:
        _, X_sample, _, y_sample = sampler_split(
            X_full, y_full, test_size=sample_size, random_state=42, stratify=y_full
        )
    else:
        X_sample = X_full.sample(n=sample_size, random_state=42)
        y_sample = y_full.loc[X_sample.index]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample.values, y_sample.values, test_size=0.3, random_state=42, stratify=y_sample if len(np.unique(y_sample)) > 1 else None
    )

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Angle encoding expects inputs in [0, pi]
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = angle_scaler.fit_transform(X_train_pca)
    X_test_final = angle_scaler.transform(X_test_pca)
    
    return X_train_final, X_test_final, y_train, y_test


def plot_pca(X_train_final, y_train, dataset_name, target_dir):
    plt.figure(figsize=(6, 4))
    # Two distinct colors for binary classes
    scatter = plt.scatter(X_train_final[:, 0], X_train_final[:, 1], c=y_train, cmap='coolwarm', edgecolor='k')
    plt.title(f"PCA Reduced {dataset_name} (Angle Encoded)", fontsize=12)
    plt.xlabel("Feature 1 (Angle)")
    plt.ylabel("Feature 2 (Angle)")
    
    # Adding a legend for distinct colors
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_label('Class')
    
    plt.tight_layout()
    plt.savefig(f'{target_dir}/{dataset_name}_pca_reduction.png', dpi=300)
    plt.close()


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, is_vqc=False):
    start_time = time.time()
    
    if is_vqc:
        # VQC expects one-hot encoded labels
        y_train_fit = np.eye(2)[y_train]
        model.fit(X_train, y_train_fit)
    else:
        model.fit(X_train, y_train)
        
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    
    # Re-map one-hot predictions to scalar labels
    if is_vqc and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        
    # Expert-level metric calculation handling class imbalance
    # Use binary averaging targeting the 'Bug' class (1)
    # Balanced accuracy is more informative for defect prediction than pure accuracy
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Balanced Acc": round(b_acc, 3),
        "Precision": round(p, 3),
        "Recall": round(r, 3),
        "F1-Score": round(f, 3),
        "Train Time (s)": round(train_time, 3)
    }


def generate_plots(df_results, dataset_name=None, target_dir='results_plots'):
    """Regenerate plots from the results dataframe ensuring order and exact labels."""
    
    # Enforce order
    model_order = ["Classical SVM", "VQC (Ideal)", "VQC (Noisy)", "QSVC (Ideal)", "QSVC (Noisy)"]
    
    # Convert 'Model' column to categorical with specific order to maintain x-axis ordering
    df_results['Model'] = pd.Categorical(df_results['Model'], categories=model_order, ordered=True)
    df_results = df_results.sort_values('Model').reset_index(drop=True)

    colors = ['royalblue', 'forestgreen', 'lightgreen', 'darkviolet', 'violet']
    
    title_suffix = f": {dataset_name}" if dataset_name else ""
    file_prefix = f"{dataset_name}_" if dataset_name else ""

    # 1. Balanced Accuracy Chart
    plt.figure(figsize=(10, 6))
    metric_to_plot = 'Balanced Acc' if 'Balanced Acc' in df_results.columns else 'Accuracy'
    bars = plt.bar(df_results['Model'], df_results[metric_to_plot], color=colors, edgecolor='k')
    plt.ylim(0, 1.15)
    plt.ylabel(metric_to_plot, fontsize=12)
    plt.title(f'Predictive Performance (Balanced){title_suffix}', fontsize=14)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{target_dir}/{file_prefix}accuracy.png', dpi=300)
    plt.close()

    # 2. Training Time Chart
    plt.figure(figsize=(10, 6))
    time_bars = plt.bar(df_results['Model'], df_results['Train Time (s)'], color=colors, edgecolor='k')
    plt.ylabel('Training Time (Seconds)', fontsize=12)
    plt.title(f'Computational Overhead{title_suffix}', fontsize=14)
    
    max_height = df_results['Train Time (s)'].max()
    plt.ylim(0, max_height * 1.15 if max_height > 0 else 1) 
    
    for bar in time_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_height * 0.02), f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{target_dir}/{file_prefix}times.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantum Machine Learning Comparative Study")
    parser.add_argument('--plot-only', action='store_true', help="Regenerate plots from existing CSV without retraining models")
    args = parser.parse_args()

    os.makedirs("results_plots", exist_ok=True)

    if args.plot_only:
        csv_path = 'results_plots/combined_performance_metrics.csv'
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Run without --plot-only first.")
            return
            
        print(f"Loading results from {csv_path}...")
        df_results = pd.read_csv(csv_path)
        
        # Regenerate plots for each dataset found in the CSV
        if 'Dataset' in df_results.columns:
            for d_name in df_results['Dataset'].unique():
                print(f"Regenerating plots for {d_name}...")
                target_dir = os.path.join("results_plots", d_name)
                os.makedirs(target_dir, exist_ok=True)
                df_subset = df_results[df_results['Dataset'] == d_name]
                generate_plots(df_subset, d_name, target_dir)
        else:
            generate_plots(df_results)
        
        print("Done.")
        return

    print("Starting Multi-Dataset Comparative Study...\n")
    
    # List of datasets to process
    dataset_dir = "dataset"
    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    dataset_files.sort()

    all_results = []
    algorithm_globals.random_seed = 42

    for dataset_file in dataset_files:
        dataset_name = dataset_file.replace('.csv', '')
        file_path = os.path.join(dataset_dir, dataset_file)
        
        # Expert organization: Create dataset-specific subfolder
        target_dir = os.path.join("results_plots", dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n>>> Processing Dataset: {dataset_name} <<<")
        
        # 1. Data Preprocessing
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
        plot_pca(X_train, y_train, dataset_name, target_dir)

        # 2. Classical Model
        print(f"[{dataset_name}] Training Classical SVM...")
        svm_model = SVC(kernel='rbf', gamma='scale', random_state=42)
        res_svm = evaluate_model(svm_model, X_train, y_train, X_test, y_test, "Classical SVM")
        res_svm['Dataset'] = dataset_name

        # 3. Quantum Environment Setup
        n_qubits = X_train.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2, entanglement='linear')
        optimizer = COBYLA(maxiter=100)
        ideal_sampler = AerSamplerV2(default_shots=1024)
        
        noise_model = NoiseModel()
        error_1q = depolarizing_error(0.02, 1)
        error_2q = depolarizing_error(0.08, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        noisy_sampler = AerSamplerV2(default_shots=1024, options={'backend_options': {'noise_model': noise_model}})
        pm = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())

        # 4. Quantum Models Training
        print(f"[{dataset_name}] Training VQC (Ideal)...")
        vqc_ideal = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=ideal_sampler, pass_manager=pm)
        res_vqc_ideal = evaluate_model(vqc_ideal, X_train, y_train, X_test, y_test, "VQC (Ideal)", is_vqc=True)
        res_vqc_ideal['Dataset'] = dataset_name

        print(f"[{dataset_name}] Training VQC (Noisy)...")
        vqc_noisy = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=noisy_sampler, pass_manager=pm)
        res_vqc_noisy = evaluate_model(vqc_noisy, X_train, y_train, X_test, y_test, "VQC (Noisy)", is_vqc=True)
        res_vqc_noisy['Dataset'] = dataset_name

        print(f"[{dataset_name}] Training QSVC (Ideal)...")
        kernel_ideal = FidelityQuantumKernel(fidelity=ComputeUncompute(sampler=ideal_sampler, transpiler=pm), feature_map=feature_map)
        qsvc_ideal = QSVC(quantum_kernel=kernel_ideal)
        res_qsvc_ideal = evaluate_model(qsvc_ideal, X_train, y_train, X_test, y_test, "QSVC (Ideal)")
        res_qsvc_ideal['Dataset'] = dataset_name

        print(f"[{dataset_name}] Training QSVC (Noisy)...")
        kernel_noisy = FidelityQuantumKernel(fidelity=ComputeUncompute(sampler=noisy_sampler, transpiler=pm), feature_map=feature_map)
        qsvc_noisy = QSVC(quantum_kernel=kernel_noisy)
        res_qsvc_noisy = evaluate_model(qsvc_noisy, X_train, y_train, X_test, y_test, "QSVC (Noisy)")
        res_qsvc_noisy['Dataset'] = dataset_name

        # Aggregate Results for this Dataset
        dataset_results = [res_svm, res_vqc_ideal, res_vqc_noisy, res_qsvc_ideal, res_qsvc_noisy]
        df_dataset = pd.DataFrame(dataset_results)
        generate_plots(df_dataset, dataset_name, target_dir)
        
        # Save local metrics CSV for this dataset
        df_dataset.to_csv(os.path.join(target_dir, f"{dataset_name}_metrics.csv"), index=False)
        
        all_results.extend(dataset_results)
        print(f"Finished {dataset_name}.")

    # 5. Final Result Aggregation and Logging
    df_all = pd.DataFrame(all_results)
    
    print("\n============================== CONSOLIDATED PERFORMANCE ==============================")
    print(df_all.sort_values(['Dataset', 'Model']).to_string(index=False))
    print("====================================================================================")
    
    df_all.to_csv('results_plots/combined_performance_metrics.csv', index=False)
    print("\nAll datasets processed. Master report saved to 'results_plots/combined_performance_metrics.csv'.")


if __name__ == "__main__":
    main()
