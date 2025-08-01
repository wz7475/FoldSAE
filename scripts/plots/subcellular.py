import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_lambda_from_path(path):
    # Match after last underscore: -.2, .8, -3, 3.4, 0, etc.
    match = re.search(r'_([\-]?\d*\.\d+|[\-]?\.\d+|[\-]?\d+)$', path)
    val = None
    if match:
        val = match.group(1)
        # Normalize: .8 -> 0.8, -.2 -> -0.2
        if val.startswith('.'):
            val = '0' + val
        elif val.startswith('-.'):
            val = '-0' + val[1:]
    if val is not None:
        try:
            fval = float(val)
            if fval.is_integer():
                return str(int(fval))
            else:
                return str(fval)
        except ValueError:
            return None
    return None

def find_lambda_dirs(input_dir):
    lambda_dirs = {}
    for root, dirs, files in os.walk(input_dir):
        for d in dirs:
            lambda_val = parse_lambda_from_path(d)
            if lambda_val is not None:
                full_path = os.path.join(root, d)
                lambda_dirs[lambda_val] = full_path
    return lambda_dirs

def get_class_counts(csv_path, all_classes):
    if not os.path.exists(csv_path):
        return 0, {cls: 0 for cls in all_classes}
    df = pd.read_csv(csv_path)
    total = len(df)
    counts = df['Subcellular Localization'].value_counts().to_dict()
    # Ensure all classes present
    return total, {cls: counts.get(cls, 0) for cls in all_classes}

def collect_all_classes(lambda_dirs):
    classes = set()
    for ldir in lambda_dirs.values():
        csv_path = os.path.join(ldir, f'classifiers/{{coeff_num}}/classifiers.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            classes.update(df['Subcellular Localization'].unique())
    return sorted(classes)

def main():
    parser = argparse.ArgumentParser(description='Multi-pane bar plot of Subcellular Localization counts by lambda.')
    parser.add_argument('--input_dir', required=True, help='Input directory (e.g., long_sweep)')
    parser.add_argument('--output_png', required=True, help='Output PNG file path')
    parser.add_argument('--ncols', type=int, default=3, help='Number of columns for subplot grid (default: 3)')
    parser.add_argument('--coeff_num', type=int, default=50, help='Number of coefficients (default: 50)')
    args = parser.parse_args()

    lambda_dirs = find_lambda_dirs(args.input_dir)
    if not lambda_dirs:
        print('No lambda directories found.')
        return

    global coeff_num
    coeff_num = args.coeff_num
    all_classes = collect_all_classes(lambda_dirs)
    if not all_classes:
        # fallback: use example classes
        all_classes = ['Cytoplasm', 'Nucleus',]

    # Collect counts for each lambda, sort by float value
    sorted_lambdas = sorted(lambda_dirs.keys(), key=lambda x: float(x))
    data = {}
    for lambda_val in sorted_lambdas:
        ldir = lambda_dirs[lambda_val]
        csv_path = os.path.join(ldir, f'classifiers/{coeff_num}/classifiers.csv')
        total, class_counts = get_class_counts(csv_path, all_classes)
        data[lambda_val] = (total, class_counts)

    n_lambdas = len(data)
    ncols = args.ncols
    nrows = (n_lambdas + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    axes = axes.flatten() if n_lambdas > 1 else [axes]
    for idx, (lambda_val, (total, counts)) in enumerate(data.items()):
        ax = axes[idx]
        bar_labels = ['Total'] + all_classes
        bar_values = [total] + [counts[cls] for cls in all_classes]
        colors = ['lightblue'] + ['blue'] * len(all_classes)
        bars = ax.bar(bar_labels, bar_values, color=colors)
        ax.set_title(f'Lambda {lambda_val}')
        ax.set_ylabel('Count')
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
    # Hide unused axes
    for ax in axes[n_lambdas:]:
        ax.axis('off')
    axes[-1].set_xlabel('Subcellular Localization')
    plt.suptitle(f'Number of subcellular classes for {coeff_num} coefficients', fontsize=32)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output_png)
    print(f'Saved plot to {args.output_png}')

if __name__ == '__main__':
    main()
