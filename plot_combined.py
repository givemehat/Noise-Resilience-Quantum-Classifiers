import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_revision/raw_results.csv')

# 1. Create a formatted Summary Table (Mean ± Std) for the Paper
summary = df.groupby(['Dataset', 'Noise', 'Model']).agg({
    'Accuracy': ['mean', 'std'],
    'F1': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std']
})
summary.columns = ['_'.join(col) for col in summary.columns.values]

formatted_table = pd.DataFrame()
for metric in ['Accuracy', 'F1', 'Precision', 'Recall']:
    formatted_table[metric] = summary.apply(
        lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" if pd.notnull(row[f'{metric}_std']) else f"{row[f'{metric}_mean']:.3f}", 
        axis=1
    )

formatted_table = formatted_table.reset_index()
formatted_table.to_csv('results_revision/Paper_Summary_Table.csv', index=False)

# 2. Create Combined Bar Plots
sns.set_theme(style="whitegrid")

# F1 Score Combined Plot
g_f1 = sns.catplot(
    data=df, kind="bar",
    x="Dataset", y="F1", hue="Model", col="Noise",
    capsize=.1, errorbar="sd", height=5, aspect=1.2,
    palette="viridis"
)
g_f1.set_axis_labels("", "F1 Score")
g_f1.set_titles("{col_name} Noise")
g_f1.set_xticklabels(rotation=45)
plt.tight_layout()
g_f1.savefig('results_revision/plots/Combined_F1_Scores_Paper.png', dpi=300)
plt.close()

# Accuracy Combined Plot
g_acc = sns.catplot(
    data=df, kind="bar",
    x="Dataset", y="Accuracy", hue="Model", col="Noise",
    capsize=.1, errorbar="sd", height=5, aspect=1.2,
    palette="viridis"
)
g_acc.set_axis_labels("", "Accuracy")
g_acc.set_titles("{col_name} Noise")
g_acc.set_xticklabels(rotation=45)
plt.tight_layout()
g_acc.savefig('results_revision/plots/Combined_Accuracy_Paper.png', dpi=300)
plt.close()

print("Combined summaries and plots generated successfully!")
