import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results_revision/raw_results.csv')

# Increase base font size for better readability in paper
sns.set_theme(style="whitegrid", font_scale=1.3)

for metric in ['Accuracy', 'F1']:
    # Use larger height and aspect ratio to give bars more room
    g = sns.catplot(
        data=df, kind="bar",
        x="Dataset", y=metric, hue="Model", col="Noise",
        capsize=.1, errorbar="sd", height=6.5, aspect=1.2,
        palette="viridis", legend_out=True
    )
    
    g.set_axis_labels("", f"{metric} Score")
    g.set_titles("Noise Level: {col_name}", size=16, pad=10)
    
    # Properly rotate x-labels and align to the right to avoid overlap
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            
    # Move legend slightly to the right to avoid squeezing the last plot
    sns.move_legend(g, "center left", bbox_to_anchor=(1.02, 0.5))
    
    # Adjust spacing between subplots
    g.fig.subplots_adjust(wspace=0.15)
    
    # Save with tight bounding box so nothing gets cropped
    g.savefig(f'results_revision/plots/Combined_{metric}_Paper.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Cleaned up overlapping text and generated high-quality plots!")
