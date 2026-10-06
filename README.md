# Noise Resilience and Parameter Efficiency of Quantum Classifiers: A Comparative Analysis Using IBM Qiskit 2.5.2

**Authors:** Rajnish Singh, Utkrisht Patel, Rakesh Kumar  
*International Journal of Computational Intelligence and Applications*

This repository contains the official code, datasets, and reproducibility guide for our comparative study evaluating the resilience of Quantum Support Vector Classifiers (QSVC) against Variational Quantum Classifiers (VQC) under realistic hardware noise conditions.

---

## 1. Abstract
Quantum machine learning (QML) models are expected to do well on high-dimensional classification tasks, but there are few benchmarks that compare them with classical models under realistic noise. In this study we compare a variational quantum classifier (VQC), trained once with COBYLA and once with SPSA, and a quantum support vector classifier (QSVC) with a classical RBF-SVM. The quantum models were implemented in IBM Qiskit 2.5.2 and run without noise and under two levels of depolarizing noise (low stress and high stress). We used five binary datasets (Iris, make_moons, make_circles, Breast Cancer and Wine). Each was reduced to two features with principal component analysis (PCA) and encoded in a two-qubit ZZFeatureMap circuit. All results are the mean and standard deviation over five random seeds. 

The RBF-SVM was the most accurate model on every dataset (0.893 to 0.993). Among the quantum models, the QSVC was the most accurate on four of the five datasets, and the VQC trained with SPSA was the best quantum model on make_circles. Noise changed the accuracy only a little. In all 30 model-dataset-noise comparisons the change was smaller than the seed-to-seed standard deviation, so we cannot say that either architecture is more robust to noise. Whether precision or recall was higher depended on the dataset. The QSVC had higher recall than precision on Breast Cancer, Wine and make_circles, and slightly higher precision than recall on Iris and make_moons. After transpilation, the VQC circuit (depth 15) and the QSVC kernel circuit (depth 18) are of similar size, with six CNOT gates each. We found no quantum advantage on these tasks. The experiments do not test parameter efficiency, so the parameter counts are given for reference only.

---

## 2. Research Questions Addressed
This codebase is designed to answer three specific research questions (RQs):
* **RQ1:** What is the difference in performance of variational and kernel-based quantum classifiers under depolarizing noise across various evaluation metrics?
* **RQ2:** Is there a classification asymmetry in terms of precision and recall that is not captured by aggregate accuracy for quantum classifiers?
* **RQ3:** How do the numbers of trainable parameters of the quantum models compare with those of classical models, and do these experiments support any claim of parameter efficiency?

---

## 3. Repository Structure

```text
Noise-Resilience-Quantum-Classifiers/
│
├── README.md                      # This comprehensive reproducibility guide
├── requirements.txt               # Exact pinned dependencies for Qiskit 2.5.2 environment
├── .gitignore                     # Git ignore definitions
│
├── robustness_study.py            # Main monolithic script for end-to-end experiment execution
├── plot_combined.py               # Script for aggregating CSVs and plotting the results
│
├── dataset/                       # Physical CSV exports of the 5 exact dataset splits used
│   ├── Binary_Iris.csv
│   ├── Breast_Cancer.csv
│   ├── Make_Circles.csv
│   ├── Make_Moons.csv
│   └── Wine.csv
│
└── results_revision/              # Final output directory for all metrics and visualizations
    ├── Oo_Paper_Summary_Table.csv # Main Accuracy/Precision/Recall/F1 table (Mean ± Std)
    ├── Oo_circuit_metrics.csv     # Transpiled circuit metrics (Depth, 1Q Gates, CNOTs)
    ├── per_seed_results.csv       # Granular data recording metrics for every individual seed
    ├── raw_results.csv            # Standard raw output file populated by the main script
    └── plots/                     # Output directory for seaborn visualizations
        ├── Oo_Combined_Accuracy_Landscape.png
        ├── Oo_Combined_Accuracy_Vertical.png
        ├── Oo_Combined_F1_Landscape.png
        └── Oo_Combined_F1_Vertical.png
```

---

## 4. Environment & Installation

To ensure exact reproducibility, it is highly recommended to run this code inside an isolated virtual environment matching the exact dependency tree used in the paper.

**Requirements:**
* Python 3.10+
* Mac / Linux / Windows

**Setup Commands:**
```bash
# Clone the repository
git clone https://github.com/givemehat/Noise-Resilience-Quantum-Classifiers.git
cd Noise-Resilience-Quantum-Classifiers

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the exact dependencies (Qiskit 2.5.2, Aer 0.17.2, scikit-learn 1.5.2, etc.)
pip install -r requirements.txt
```

---

## 5. Experimental Methodology

### Datasets & Preprocessing (Algorithm 1)
We evaluated the models on 5 standard binary datasets. While physical `.csv` exports are available in the `dataset/` folder for manual inspection, the primary script `robustness_study.py` loads them dynamically via `scikit-learn` to preserve exact state configurations.

**Preprocessing Pipeline:**
1. Split 70%/30% (stratified, `random_state=seed`)
2. `StandardScaler` (zero mean, unit variance)
3. `PCA (n_components=2)`
4. `MinMaxScaler([0, π])`

**Subsampling:** The *Breast Cancer* and *Wine* datasets are downsampled to exactly 100 instances using stratified random sampling to respect the simulation constraints of noisy quantum circuits.

### Quantum Models & Circuits
* **Encoding:** A 2-qubit `ZZFeatureMap` (reps=2).
* **VQC Ansatz:** `RealAmplitudes` (reps=2), providing 6 trainable parameters.
* **Optimizers:** 
  * Gradient-free: **COBYLA** (budget = 300 max iterations).
  * Stochastic gradient descent: **SPSA** (budget = 100 max iterations).
* **QSVC Kernel:** `FidelityQuantumKernel` using the `ComputeUncompute` algorithm.

### Noise Models (IBM Qiskit Aer)
Experiments test three conditions using exact gate-level depolarizing errors:
1. **Ideal (0.0):** Executed using Qiskit's exact `StatevectorSampler` (deterministic, zero shot-noise).
2. **Low Stress (p1=0.01, p2=0.04):** 1% 1-qubit depolarizing error and 4% 2-qubit depolarizing error applied to all transpiled gates (`h`, `ry`, `p`, `cx`) via `AerSamplerV2` (1024 shots).
3. **High Stress (p1=0.02, p2=0.08):** 2% 1-qubit error and 8% 2-qubit error via `AerSamplerV2` (1024 shots).

---

## 6. Reproducing the Experiments

All randomness (train/test splits, PCA initialization, Optimizer weight initialization, and Simulator shot noise) is strictly anchored using 5 global seeds (`42, 100, 2023, 777, 1234`) across Numpy, Qiskit `algorithm_globals`, and the `AerSamplerV2` `seed_simulator` option.

To run the full end-to-end simulation from scratch:

```bash
# 1. Execute the main training and evaluation script (Expected execution time: 30-45 minutes on local hardware)
python robustness_study.py

# 2. Generate aggregated tables (Mean ± Std, ddof=1) and layout plots
python plot_combined.py
```

*Note: You do not need to re-run these scripts to view the results. The official peer-reviewed outputs are already populated inside the `results_revision/` directory.*

---

## 7. Tracing Results to the Paper

A reviewer inspecting this repository can map the claims in the paper directly to the outputs in `results_revision/`:

### Table 2 & 5 (Accuracy, Precision, Recall, F1 Metrics)
* Refer to `results_revision/Oo_Paper_Summary_Table.csv` and `results_revision/per_seed_results.csv`.
* **Addressing RQ1 & RQ2:** The table mathematically verifies the classification asymmetry claim from Section 5.4: QSVC has higher recall than precision on *Breast Cancer*, *Wine*, and *make_circles*, but slightly higher precision on *Iris* and *make_moons*.

### Table 4 (Gate Metrics)
* Refer to `results_revision/Oo_circuit_metrics.csv`.
* **Addressing RQ3:** Confirms the Transpiled Depth (15 for VQC vs 18 for Kernel), 1-Qubit gate counts (16 vs 20), and CNOTs (6 vs 6), verifying that the kernel circuit is slightly deeper than the VQC circuit.

### Figures 2 & 3 (Performance Plots)
* Refer to `results_revision/plots/Oo_Combined_Accuracy_Vertical.png` and `results_revision/plots/Oo_Combined_F1_Vertical.png`.
* These images are publication-ready visualizations containing Standard Deviation error bars over the 5 random seeds, perfectly corresponding to Figures 2 and 3 in the manuscript.

---

## 8. License & Citation
*If you use this codebase in your research, please cite the corresponding paper in the International Journal of Computational Intelligence and Applications.*
