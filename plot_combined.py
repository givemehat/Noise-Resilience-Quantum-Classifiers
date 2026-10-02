import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_revision/raw_results.csv')

# --- Generate Paper Summary Table (mean +/- std with ddof=1) ---
summary = df.groupby(['Dataset', 'Noise', 'Model']).agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1': ['mean', 'std']
})
summary.columns = ['_'.join(col) for col in summary.columns.values]

formatted_table = pd.DataFrame()
formatted_table['Dataset'] = summary.index.get_level_values('Dataset')
formatted_table['Noise'] = summary.index.get_level_values('Noise')
formatted_table['Model'] = summary.index.get_level_values('Model')

for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    formatted_table[metric] = summary.apply(
        lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" if pd.notnull(row[f'{metric}_std']) else f"{row[f'{metric}_mean']:.3f}", 
        axis=1
    ).values

formatted_table.to_csv('results_revision/Oo_Paper_Summary_Table.csv', index=False)
print("Saved Oo_Paper_Summary_Table.csv")

# --- Generate Plots ---
sns.set_theme(style="whitegrid", font_scale=1.1)

for metric in ['Accuracy', 'F1']:
    # 1. LANDSCAPE VERSION (For Page-Wide Figure)
    g_wide = sns.catplot(
        data=df, kind="bar", x="Dataset", y=metric, hue="Model", col="Noise",
        capsize=.1, errorbar="sd", height=5, aspect=1.3, palette="viridis"
    )
    g_wide.set_axis_labels("", f"{metric} Score")
    g_wide.set_titles("{col_name} Noise", size=14, pad=10)
    for ax in g_wide.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    sns.move_legend(g_wide, "center left", bbox_to_anchor=(1.02, 0.5))
    g_wide.fig.subplots_adjust(wspace=0.1)
    g_wide.savefig(f'results_revision/plots/Oo_Combined_{metric}_Landscape.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. VERTICAL/PORTRAIT VERSION (For Single Column in Paper)
    g_tall = sns.catplot(
        data=df, kind="bar", x="Dataset", y=metric, hue="Model", row="Noise", # Stacked rows
        capsize=.1, errorbar="sd", height=3.5, aspect=1.8, palette="viridis"
    )
    g_tall.set_axis_labels("", f"{metric} Score")
    g_tall.set_titles("{row_name} Noise", size=14, pad=10)
    # Only rotate labels on the bottom-most plot
    for ax in g_tall.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
            
    # Move legend outside the top right
    sns.move_legend(g_tall, "center left", bbox_to_anchor=(1.02, 0.5))
    g_tall.fig.subplots_adjust(hspace=0.4)
    g_tall.savefig(f'results_revision/plots/Oo_Combined_{metric}_Vertical.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Generated both Landscape and Vertical plot versions with Oo_ prefix!")
