import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_revision/raw_results.csv')
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
    g_wide.savefig(f'results_revision/plots/Combined_{metric}_Landscape.png', dpi=300, bbox_inches='tight')
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
    g_tall.savefig(f'results_revision/plots/Combined_{metric}_Vertical.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Generated both Landscape and Vertical versions!")
