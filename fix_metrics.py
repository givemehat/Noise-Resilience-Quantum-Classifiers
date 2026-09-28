import pandas as pd
import numpy as np
import os

def check_metrics(row):
    """
    Diagnose consistency problems in a single row.
    """
    p, r, f1 = row['Precision'], row['Recall'], row['F1-Score']
    acc = row['Accuracy']
    
    issues = []
    
    # Mathematical F1 consistency check
    expected_f1 = 0.0
    if (p + r) > 0:
        expected_f1 = 2 * (p * r) / (p + r)
    
    if abs(expected_f1 - f1) > 0.001:
        issues.append(f"Inconsistent F1: Given {f1:.4f}, Calculated {expected_f1:.4f}")
    
    # Accuracy Paradox check (Majority Class Classifier Bias)
    if acc > 0.7 and r == 0.0:
        issues.append("Majority-Class Bias: High accuracy but zero recall (suggests failure to detect bugs)")
        
    return issues, expected_f1

def main():
    csv_path = 'results_plots/combined_performance_metrics.csv'
    output_path = 'results_plots/combined_performance_metrics_clean.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Data Loading
    df = pd.read_csv(csv_path)
    
    # 2. Metric consistency checks
    updated_f1 = []
    flags = []
    
    print("--- Metric Diagnosis Report ---\n")
    
    for idx, row in df.iterrows():
        issues, recomputed_f1 = check_metrics(row)
        updated_f1.append(recomputed_f1)
        
        if issues:
            flag_str = "; ".join(issues)
            print(f"Row {idx} ({row['Dataset']} - {row['Model']}):")
            print(f"  Issues: {flag_str}")
            flags.append(flag_str)
        else:
            flags.append("")
            
    # 3. Apply fixes
    # Overwrite F1-Score with consistent value
    df['F1-Score'] = updated_f1
    
    # 4. Formatting fixes
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df[metric_cols] = df[metric_cols].round(3)
    df["Train Time (s)"] = df["Train Time (s)"].round(3)
    
    # Define custom sort order for Models
    model_order = ["Classical SVM", "VQC (Ideal)", "VQC (Noisy)", "QSVC (Ideal)", "QSVC (Noisy)"]
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    
    # Sort by Dataset, then Model
    df = df.sort_values(['Dataset', 'Model']).reset_index(drop=True)
    
    # 5. Output
    print("\n\n--- Cleaned Summary Table ---")
    print("Combined performance metrics across all datasets and models.")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nCorrected table saved to: {output_path}")
    
    # Final head for inspection
    print("\nDataFrame Head (Final Formatting):")
    print(df.head())

if __name__ == "__main__":
    main()
