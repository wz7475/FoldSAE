import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def create_plot(input_dir, output_png):
    data = []
    for dirname in os.listdir(input_dir):
        if dirname.startswith('output_blocks_'):
            block_num = int(dirname.split('_')[-1])
            csv_path = os.path.join(input_dir, dirname, 'preds.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['block'] = block_num
                data.append(df)
    if not data:
        print("No preds.csv files found in the specified input_dir.")
        return

    combined_df = pd.concat(data, ignore_index=True)
    kept_classes = ['Cytoplasm', 'Nucleus', 'Extra - cellular']
    filtered_df = combined_df[combined_df['Subcellular Localization'].isin(kept_classes)]
    total_counts = filtered_df.groupby('block').size().reset_index(name='total_count')
    class_counts = filtered_df.groupby(['block', 'Subcellular Localization']).size().reset_index(name='class_count')
    merged = pd.merge(class_counts, total_counts, on='block')
    merged['ratio'] = merged['class_count'] / merged['total_count']

    def color_func(block):
        return 'red' if block == 4 else 'green'
    merged['color'] = merged['block'].apply(color_func)

    g = sns.FacetGrid(
        merged,
        row='Subcellular Localization',
        sharex=True,
        sharey=True,
        height=3,
        aspect=4
    )

    def barplot_with_colors(data, **kwargs):
        plt.bar(
            data['block'].astype(str),
            data['ratio'],
            color=data['color']
        )
    g.map_dataframe(barplot_with_colors)

    g.set_axis_labels('Number of Ablated Block (-1 means no block ablated)', 'Ratio of Samples')
    g.set_titles(row_template='{row_name}')

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create plot of Subcellular Localization ratio per ablated block')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory containing output_blocks_* folders')
    parser.add_argument('--output_png', type=str, required=True, help='Path to output PNG file')
    args = parser.parse_args()
    create_plot(args.input_dir, args.output_png)
