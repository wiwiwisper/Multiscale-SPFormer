#!/usr/bin/env python3
"""
Parse Evaluation_metrics.log and plot training curves
Usage: python tools/plot_metrics.py ./exps/myplants/Evaluation_metrics.log
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_metrics_log(log_file):
    """Parse the Evaluation_metrics.log file"""
    epochs = []
    timestamps = []
    metrics = {
        'stem': {'AP': [], 'AP_50%': [], 'AP_25%': [], 'AR': [], 'RC_50%': [], 'RC_25%': []},
        'leaf': {'AP': [], 'AP_50%': [], 'AP_25%': [], 'AR': [], 'RC_50%': [], 'RC_25%': []},
        'average': {'AP': [], 'AP_50%': [], 'AP_25%': [], 'AR': [], 'RC_50%': [], 'RC_25%': []}
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for epoch header
        if line.startswith('Epoch:'):
            # Parse epoch and timestamp
            epoch_match = re.search(r'Epoch:\s*(\d+)', line)
            time_match = re.search(r'Time:\s*([\d\-: ]+)', line)

            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
                if time_match:
                    timestamps.append(time_match.group(1))
                else:
                    timestamps.append('')

            # Skip to metrics table
            while i < len(lines) and 'what' not in lines[i]:
                i += 1
            i += 2  # Skip header and separator

            # Parse metrics for each class
            for class_name in ['stem', 'leaf', 'average']:
                if i >= len(lines):
                    break
                line = lines[i].strip()
                if line.startswith(class_name) or (class_name == 'average' and line.startswith('average')):
                    # Extract metrics using regex
                    # Format: stem           :   0.985   1.000   1.000   0.987   1.000   1.000
                    values = re.findall(r'\d+\.\d+', line)
                    if len(values) == 6:
                        metrics[class_name]['AP'].append(float(values[0]))
                        metrics[class_name]['AP_50%'].append(float(values[1]))
                        metrics[class_name]['AP_25%'].append(float(values[2]))
                        metrics[class_name]['AR'].append(float(values[3]))
                        metrics[class_name]['RC_50%'].append(float(values[4]))
                        metrics[class_name]['RC_25%'].append(float(values[5]))
                i += 1
        i += 1

    return epochs, timestamps, metrics


def plot_metrics(epochs, metrics, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Evaluation Metrics', fontsize=16, fontweight='bold')

    metric_names = ['AP', 'AP_50%', 'AP_25%', 'AR', 'RC_50%', 'RC_25%']

    for idx, metric_name in enumerate(metric_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Plot each class
        for class_name in ['stem', 'leaf', 'average']:
            if len(metrics[class_name][metric_name]) > 0:
                ax.plot(epochs, metrics[class_name][metric_name],
                       marker='o', label=class_name, linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} over Epochs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    else:
        plt.show()


def print_summary(epochs, metrics):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)
    print(f"Total Epochs Evaluated: {len(epochs)}")
    print(f"Epochs: {epochs}")
    print("\n" + "-"*80)

    for class_name in ['stem', 'leaf', 'average']:
        print(f"\n{class_name.upper()}:")
        for metric_name in ['AP', 'AP_50%', 'AP_25%']:
            values = metrics[class_name][metric_name]
            if len(values) > 0:
                print(f"  {metric_name:10s}: Best={max(values):.4f} (Epoch {epochs[values.index(max(values))]}), "
                      f"Latest={values[-1]:.4f}, Mean={np.mean(values):.4f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Parse and plot evaluation metrics')
    parser.add_argument('log_file', type=str, help='Path to Evaluation_metrics.log')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot (e.g., metrics_plot.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Only print summary without plotting')
    args = parser.parse_args()

    # Parse log file
    epochs, timestamps, metrics = parse_metrics_log(args.log_file)

    if len(epochs) == 0:
        print("No metrics found in the log file!")
        return

    # Print summary
    print_summary(epochs, metrics)

    # Plot metrics
    if not args.no_plot:
        if args.output is None:
            # Auto-generate output path
            import os
            log_dir = os.path.dirname(args.log_file)
            args.output = os.path.join(log_dir, 'metrics_plot.png')

        plot_metrics(epochs, metrics, args.output)


if __name__ == '__main__':
    main()
