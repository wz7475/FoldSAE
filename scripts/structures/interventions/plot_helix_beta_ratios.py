#!/usr/bin/env python3
"""
Generate plots for helix to beta sheet ratios analysis.
Creates separate plots for each threshold and class combination.
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def load_results(results_file):
    """Load the analysis results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_plots(results, output_dir):
    """Create plots for each threshold and class combination."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all thresholds and classes
    thresholds = sorted(results.keys(), key=float)
    
    for threshold in thresholds:
        print(f"Creating plots for threshold {threshold}")
        
        # Get all classes for this threshold
        classes = list(results[threshold].keys())
        
        for class_name in classes:
            print(f"  Creating plot for class {class_name}")
            
            # Extract data for this threshold and class
            class_data = results[threshold][class_name]
            
            # Sort lambda values numerically
            lambda_values = sorted(class_data.keys(), key=float)
            
            # Extract helix and beta ratios
            helix_ratios = []
            beta_ratios = []
            
            for lambda_val in lambda_values:
                entry = class_data[lambda_val]
                # Support both 3-value and 4-value entries
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 3:
                        helix_count, beta_count, total_residues = entry[0], entry[1], entry[2]
                    else:
                        # Fallback if malformed
                        helix_count, beta_count, total_residues = 0, 0, 0
                else:
                    helix_count, beta_count, total_residues = 0, 0, 0
                
                if total_residues > 0:
                    helix_ratio = helix_count / total_residues
                    beta_ratio = beta_count / total_residues
                else:
                    helix_ratio = 0
                    beta_ratio = 0
                
                helix_ratios.append(helix_ratio)
                beta_ratios.append(beta_ratio)
            
            # Convert lambda values to float for plotting
            lambda_values_float = [float(x) for x in lambda_values]
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            # Plot only the helix series (skip beta)
            plt.plot(lambda_values_float, helix_ratios, 'o-', label='Helix / Total Residues', 
                    color='red', linewidth=2, markersize=6)
            # (beta series intentionally skipped)
            
            # Customize the plot
            plt.xlabel('Lambda Values', fontsize=12)
            plt.ylabel('Ratio to Total Residues', fontsize=12)
            plt.title(f'Helix Ratios - Threshold: {threshold}, Class: {class_name}', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Set y-axis limits to better show the data (use helix ratios only)
            if helix_ratios:
                y_min = min(helix_ratios) * 0.9
                y_max = max(helix_ratios) * 1.1
                # Guard against zero range
                if y_max == y_min:
                    y_min = max(0, y_min - 0.05)
                    y_max = y_min + 0.1
                plt.ylim(y_min, y_max)
            
            # Add some padding to x-axis
            if len(lambda_values_float) > 1:
                x_range = max(lambda_values_float) - min(lambda_values_float)
                plt.xlim(min(lambda_values_float) - x_range * 0.05, 
                        max(lambda_values_float) + x_range * 0.05)
            
            # Save the plot
            filename = f"helix_ratios_thr_{threshold}_class_{class_name}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {filepath}")


def create_summary_plot(results, output_dir):
    """Create a summary plot with all thresholds and classes."""
    
    print("Creating summary plot...")
    
    # Create a large subplot grid
    thresholds = sorted(results.keys(), key=float)
    classes = set()
    for threshold in thresholds:
        classes.update(results[threshold].keys())
    # If a 'beta' class exists (case-insensitive), use only that single pane/column
    beta_key = None
    for c in sorted(classes):
        if c.lower() == 'beta':
            beta_key = c
            break
    if beta_key:
        classes = [beta_key]
    else:
        classes = sorted(classes)
    
    n_thresholds = len(thresholds)
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_thresholds, n_classes, figsize=(5*n_classes, 4*n_thresholds))
    
    # Ensure axes is a 2D array shaped (n_thresholds, n_classes)
    axes = np.array(axes)
    axes = axes.reshape(n_thresholds, n_classes)
    
    for i, threshold in enumerate(thresholds):
        for j, class_name in enumerate(classes):
            ax = axes[i, j]
            
            if class_name in results[threshold]:
                class_data = results[threshold][class_name]
                lambda_values = sorted(class_data.keys(), key=float)
                
                helix_ratios = []
                
                for lambda_val in lambda_values:
                    entry = class_data[lambda_val]
                    # Support both 3-value and 4-value entries
                    if isinstance(entry, (list, tuple)):
                        if len(entry) >= 3:
                            helix_count, _, total_residues = entry[0], entry[1], entry[2]
                        else:
                            helix_count, total_residues = 0, 0
                    else:
                        helix_count, total_residues = 0, 0
                    
                    if total_residues > 0:
                        helix_ratio = helix_count / total_residues
                    else:
                        helix_ratio = 0
                    
                    helix_ratios.append(helix_ratio)
                
                lambda_values_float = [float(x) for x in lambda_values]
                
                # Plot only helix series on summary grid
                ax.plot(lambda_values_float, helix_ratios, 'o-', 
                       color='red', linewidth=2, markersize=4)
                # (beta series intentionally skipped)
                
                # Title and labels
                ax.set_title('Ratio: alpha helices / total residues', fontsize=15)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                if i == n_thresholds - 1:  # Bottom row
                    ax.set_xlabel('Lambda', fontsize=10)
                # if j == 0:  # Left column
                #     ax.set_ylabel('Ratio', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Thr: {threshold}, Class: {class_name}', fontsize=10)
    
    plt.tight_layout()
    summary_filepath = os.path.join(output_dir, "helix_ratios_summary.png")
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {summary_filepath}")


def create_pdb_counts_plot(results, output_dir):
    """Create a plot of PDB file counts vs lambda for each threshold/class combination."""
    print("Creating PDB counts plot...")

    os.makedirs(output_dir, exist_ok=True)

    thresholds = sorted(results.keys(), key=float)

    plt.figure(figsize=(12, 7))

    for threshold in thresholds:
        classes = list(results[threshold].keys())
        for class_name in classes:
            class_data = results[threshold][class_name]
            lambda_values = sorted(class_data.keys(), key=float)

            lambda_values_float = []
            pdb_counts = []

            for lambda_val in lambda_values:
                entry = class_data[lambda_val]
                if isinstance(entry, (list, tuple)) and len(entry) >= 4:
                    count_pdb = entry[3]
                else:
                    # If older results without counts, skip plotting for this series
                    count_pdb = None

                if count_pdb is not None:
                    lambda_values_float.append(float(lambda_val))
                    pdb_counts.append(count_pdb)

            if lambda_values_float:
                plt.plot(
                    lambda_values_float,
                    pdb_counts,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=f"Thr {threshold} - {class_name}"
                )

    plt.xlabel('Lambda Values', fontsize=12)
    plt.ylabel('Count of PDB files', fontsize=12)
    plt.title('PDB file counts vs Lambda per Threshold/Class', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, ncol=2)

    # Add some padding to x-axis if there are any points
    # Determine global x range from plotted lines
    all_x = []
    for line in plt.gca().get_lines():
        all_x.extend(line.get_xdata())
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        if x_max > x_min:
            x_range = x_max - x_min
            plt.xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    filepath = os.path.join(output_dir, "pdb_counts_over_lambda.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PDB counts plot saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots for helix to beta sheet ratios analysis')
    parser.add_argument('--results_file', required=True, help='JSON file containing the analysis results')
    parser.add_argument('--output_dir', required=True, help='Directory to save the plots')
    parser.add_argument('--summary', action='store_true', help='Also create a summary plot with all data')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    print(f"Creating individual plots in: {args.output_dir}")
    create_plots(results, args.output_dir)
    
    if args.summary:
        create_summary_plot(results, args.output_dir)

    # Always create PDB counts plot if counts are available in results
    create_pdb_counts_plot(results, args.output_dir)
    
    print("Plot generation complete!")


if __name__ == "__main__":
    main()

