# QML SVM Robustness Study

This project contains the codebase for evaluating the resilience of Quantum Support Vector Classifiers (QSVC) against Variational Quantum Classifiers (VQC) under hardware noise.

## Updates (Addressing Reviewer Comments)
The primary script `robustness_study.py` has been completely rewritten to strictly adhere to the reviewer guidelines:
1. **Statistical Rigor**: Experiments now run across **5 independent random seeds**, plotting the Mean ± Standard Deviation.
2. **Noise Levels**: Incorporates 3 levels: Ideal (0.0), Low-Stress (0.01/0.04), and High-Stress (0.02/0.08).
3. **Harder Datasets**: Evaluates on `Binary Iris`, `Make Moons`, and `Breast Cancer Wisconsin`.
4. **Optimizers**: Trains VQC using both **COBYLA (budget increased to 300)** and **SPSA** to directly compare shot-noise aware optimizers vs limited budget effects.
5. **Circuit Hardware Metrics**: Computes exact Depth, 1-qubit gates, and CNOTs for both architectures after transpilation.
6. **Automated Plotting**: Generates bar charts with standard deviation error bars automatically in the `results_revision/plots/` folder.

## Usage
Run the unified script:
```bash
python robustness_study.py
```
Outputs are saved in `results_revision/`.
