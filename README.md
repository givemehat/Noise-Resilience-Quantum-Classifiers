# Noise Resilience and Parameter Efficiency of Quantum Classifiers

This project contains the codebase for evaluating the resilience of Quantum Support Vector Classifiers (QSVC) against Variational Quantum Classifiers (VQC) under hardware noise.

## Abstract
Quantum machine learning (QML) models are expected to do well on high-dimensional classification tasks, but there are few benchmarks that compare them with classical models under realistic noise. In this study we compare a variational quantum classifier (VQC), trained once with COBYLA and once with SPSA, and a quantum support vector classifier (QSVC) with a classical RBF-SVM. The quantum models were implemented in IBM Qiskit 2.5.2 and run without noise and under two levels of depolarizing noise (low stress and high stress). We used five binary datasets (Iris, make_moons, make_circles, Breast Cancer and Wine). Each was reduced to two features with principal component analysis (PCA) and encoded in a two-qubit ZZFeatureMap circuit. All results are the mean and standard deviation over five random seeds. 

The RBF-SVM was the most accurate model on every dataset (0.893 to 0.993). Among the quantum models, the QSVC was the most accurate on four of the five datasets, and the VQC trained with SPSA was the best quantum model on make_circles. Noise changed the accuracy only a little. In all 30 model-dataset-noise comparisons the change was smaller than the seed-to-seed standard deviation, so we cannot say that either architecture is more robust to noise. Whether precision or recall was higher depended on the dataset. The QSVC had higher recall than precision on Breast Cancer, Wine and make_circles, and slightly higher precision than recall on Iris and make_moons. After transpilation, the VQC circuit (depth 15) and the QSVC kernel circuit (depth 18) are of similar size, with six CNOT gates each. We found no quantum advantage on these tasks. The experiments do not test parameter efficiency, so the parameter counts are given for reference only.

## Experimental Setup
1. **Datasets**: 5 binary datasets (`Iris`, `make_moons`, `make_circles`, `Breast Cancer`, `Wine`).
2. **Models**: VQC (COBYLA), VQC (SPSA), QSVC, and Classical RBF-SVM.
3. **Statistical Rigor**: 5 independent random seeds (Mean ± Standard Deviation).
4. **Noise Levels**: Ideal (0.0), Low-Stress, and High-Stress depolarizing noise (including precise p-gate noise tracking).
5. **Framework**: Qiskit 2.5.2, Aer 0.17.2, QML 0.9.1.

## Usage
Run the noisy simulation protocol:
```bash
python robustness_study.py
python plot_combined.py
```
Final outputs, plots, and tables are saved in `results_revision/`.
