import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import re

def parse_enzymes_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    records = []
    for line in lines:
        parts = line.strip().split(',')
        enzyme_classes = set()
        for p in parts[1:]:
            match = re.match(r'EC:(\d+)', p)
            if match:
                enzyme_classes.add(match.group(1))
        records.append(list(enzyme_classes))
    print(f"parsing {file_path}")
    return records

def enzyme_ratios(input_dir):
    block_dirs = [d for d in os.listdir(input_dir) if d.startswith('output_blocks_')]
    result = []
    for dirname in block_dirs:
        block_num = int(dirname.split('_')[-1])
        csv_path = os.path.join(input_dir, dirname, 'enzymes.csv')
        if os.path.exists(csv_path):
            records = parse_enzymes_csv(csv_path)
            total_samples = len(records)
            class_counter = {}
            for enzyme_classes in records:
                for ec in set(enzyme_classes):
                    class_counter[ec] = class_counter.get(ec, 0) + 1
            for ec, count in class_counter.items():
                ratio = count / total_samples if total_samples > 0 else 0
                result.append({'block': block_num, 'enzyme_class': ec, 'ratio': ratio})
    return pd.DataFrame(result)

def create_plot(input_dir, output_png):
    df = enzyme_ratios(input_dir)
    df = df[
        df["enzyme_class"].isin([str(i) for i in  [1, 2, 3, 4, 5, 6]])
    ]
    if df.empty:
        print("No enzymes.csv files found or no data extracted.")
        return

    # Define colors: all blocks green except block 4 which is red
    def color_func(block):
        return 'red' if block == 4 else 'limegreen'
    df['color'] = df['block'].apply(color_func)

    # Sort for consistent plotting
    df = df.sort_values(['enzyme_class', 'block'])

    # Create FacetGrid with shared y-axis
    g = sns.FacetGrid(
        df,
        row='enzyme_class',
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
    g.set_titles(row_template='Enzymatic class {row_name}')

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create plot of enzymatic class ratios per ablated block')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory containing output_blocks_* folders')
    parser.add_argument('--output_png', type=str, required=True, help='Path to output PNG file')
    args = parser.parse_args()
    create_plot(args.input_dir, args.output_png)
