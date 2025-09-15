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
                helix_count, beta_count, total_residues = class_data[lambda_val]
                
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
            
            # Plot both series
            plt.plot(lambda_values_float, helix_ratios, 'o-', label='Alpha Helix / Total Residues', 
                    color='red', linewidth=2, markersize=6)
            plt.plot(lambda_values_float, beta_ratios, 's-', label='Beta Sheet / Total Residues', 
                    color='blue', linewidth=2, markersize=6)
            
            # Customize the plot
            plt.xlabel('Lambda Values', fontsize=12)
            plt.ylabel('Ratio to Total Residues', fontsize=12)
            plt.title(f'Helix/Beta Ratios - Threshold: {threshold}, Class: {class_name}', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Set y-axis limits to better show the data
            all_ratios = helix_ratios + beta_ratios
            if all_ratios:
                y_min = min(all_ratios) * 0.9
                y_max = max(all_ratios) * 1.1
                plt.ylim(y_min, y_max)
            
            # Add some padding to x-axis
            if len(lambda_values_float) > 1:
                x_range = max(lambda_values_float) - min(lambda_values_float)
                plt.xlim(min(lambda_values_float) - x_range * 0.05, 
                        max(lambda_values_float) + x_range * 0.05)
            
            # Save the plot
            filename = f"helix_beta_ratios_thr_{threshold}_class_{class_name}.png"
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
    classes = sorted(classes)
    
    n_thresholds = len(thresholds)
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_thresholds, n_classes, figsize=(5*n_classes, 4*n_thresholds))
    
    # Handle case where we have only one threshold or one class
    if n_thresholds == 1:
        axes = axes.reshape(1, -1)
    if n_classes == 1:
        axes = axes.reshape(-1, 1)
    
    for i, threshold in enumerate(thresholds):
        for j, class_name in enumerate(classes):
            ax = axes[i, j]
            
            if class_name in results[threshold]:
                class_data = results[threshold][class_name]
                lambda_values = sorted(class_data.keys(), key=float)
                
                helix_ratios = []
                beta_ratios = []
                
                for lambda_val in lambda_values:
                    helix_count, beta_count, total_residues = class_data[lambda_val]
                    
                    if total_residues > 0:
                        helix_ratio = helix_count / total_residues
                        beta_ratio = beta_count / total_residues
                    else:
                        helix_ratio = 0
                        beta_ratio = 0
                    
                    helix_ratios.append(helix_ratio)
                    beta_ratios.append(beta_ratio)
                
                lambda_values_float = [float(x) for x in lambda_values]
                
                ax.plot(lambda_values_float, helix_ratios, 'o-', label='Helix', 
                       color='red', linewidth=2, markersize=4)
                ax.plot(lambda_values_float, beta_ratios, 's-', label='Beta', 
                       color='blue', linewidth=2, markersize=4)
                
                ax.set_title(f'Thr: {threshold}, Class: {class_name}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                if i == n_thresholds - 1:  # Bottom row
                    ax.set_xlabel('Lambda', fontsize=10)
                if j == 0:  # Left column
                    ax.set_ylabel('Ratio', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Thr: {threshold}, Class: {class_name}', fontsize=10)
    
    plt.tight_layout()
    summary_filepath = os.path.join(output_dir, "helix_beta_ratios_summary.png")
    plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {summary_filepath}")


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
    
    print("Plot generation complete!")


if __name__ == "__main__":
    main()
