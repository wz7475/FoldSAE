import csv
import glob
import os
import subprocess
import time
from argparse import ArgumentParser

import wandb
import sys

import matplotlib.pyplot as plt
import pandas as pd


def prepare_command_rfdiffusion(
    input_dir,
    prefix,
    length,
    num_designs,
    python="/home/wzarzecki/miniforge3/envs/rf/bin/python",
) -> str:
    return f"""
{python} RFDiffSAE/scripts/run_inference.py \
  inference.output_prefix={input_dir}/{length}/{prefix} \
  'contigmap.contigs=[{length}-{length}]' \
  inference.num_designs={num_designs} \
  inference.final_step=1 \
  inference.use_random_suffix_for_new_design=False \
  inference.seed=1
"""


def run_rfdiffusion(command):
    tic = time.time()
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3600 * 48,  # Set appropriate timeout
    )
    print(f"elapsed time runnning RFdiffusion: {time.time() - tic}")
    return result


def log_result(wandb_handle, result):
    wandb_handle.log(
        {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def alpha_helix_to_beta_sheet(path: str):
    with open(path) as fp:
        lines = fp.readlines()
    ghi_counter = 0
    e_counter = 0
    glycine_counter = 0
    for line in lines:
        if line.startswith("STR"):
            ghi_counter += line.count("G")
            ghi_counter += line.count("H")
            ghi_counter += line.count("I")
            e_counter += line.count("E")
        if line.startswith("SEQ"):
            glycine_counter += line.count("G")
    return ghi_counter, e_counter, glycine_counter


def write_to_file(alpha_helix, beta_sheet, glycine, fp, write_header=False):
    csvwriter = csv.writer(fp)
    if write_header:
        csvwriter.writerow(["ratio", "seq_len", "alpha", "beta"])
    csvwriter.writerow(
        [
            alpha_helix / beta_sheet if beta_sheet else 1.0,
            glycine,
            alpha_helix,
            beta_sheet,
        ]
    )


def log_to_file(alpha_helix, beta_sheet, glycine, file_path):
    if os.path.exists(file_path):
        with open(file_path, "a+") as fp:
            write_to_file(alpha_helix, beta_sheet, glycine, fp)
    else:
        with open(file_path, "w") as fp:
            write_to_file(alpha_helix, beta_sheet, glycine, fp, write_header=True)


def run_stride(pdb_path: str, output_path: str, stride_path):
    subprocess.run(
        f"{stride_path} -o {pdb_path} > {output_path}",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

def plot_stats(stats_file, output_path):
    # Read stats.csv
    df = pd.read_csv(stats_file)
    # Group by seq_len and calculate mean and std
    grouped = df.groupby('seq_len').agg({
        'ratio': ['mean', 'std'],
        'alpha': ['mean', 'std'],
        'beta': ['mean', 'std']
    }).reset_index()

    seq_lens = grouped['seq_len'].astype(str)
    x = range(len(seq_lens))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left pane: mean ratio with std error bars
    axes[0].bar(x, grouped['ratio']['mean'], yerr=grouped['ratio']['std'], color='skyblue', capsize=5)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Alpha/Beta Ratio (mean ± std)')
    axes[0].set_title('Alpha/Beta Ratio by Sequence Length')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seq_lens, rotation=45)

    # Right pane: mean alpha and beta with std error bars
    width = 0.35
    axes[1].bar([i - width/2 for i in x], grouped['alpha']['mean'], width, yerr=grouped['alpha']['std'], label='Alpha', color='orange', capsize=5)
    axes[1].bar([i + width/2 for i in x], grouped['beta']['mean'], width, yerr=grouped['beta']['std'], label='Beta', color='green', capsize=5)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Count (mean ± std)')
    axes[1].set_title('Alpha and Beta Counts by Sequence Length')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seq_lens, rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main(num_designs, python, input_dir, prefix, length, stats_file_name, stride_path, plot_file_name, disable_structure_gen=False, disable_stride_eval=True):
    wandb.init()

    try:
        ########################################
        ##### 1) generate structures with RFDiffusion
        if not disable_structure_gen:
            command = prepare_command_rfdiffusion(
                input_dir, prefix, length, num_designs, python
            )
            result = run_rfdiffusion(command)
            print("executed rfdiffusion")
            if result.returncode != 0:
                sys.exit(result.returncode)
            log_result(wandb, result)
            print("logged results")
        ########################################
        ##### 2) parse results with Stride parser
        if not disable_stride_eval:
            for pdb_file in glob.glob(f"{input_dir}/**/*.pdb", recursive=True):
                stride_file = pdb_file.replace(".pdb", ".stride")
                run_stride(pdb_file, stride_file, stride_path)
                print("executed stride")
                alpha_helix, beta_sheet, glycine = alpha_helix_to_beta_sheet(stride_file)
                log_to_file(alpha_helix, beta_sheet, glycine, os.path.join(input_dir, stats_file_name))
                print("parsed strided")
            # 3) plot stats after parsing
            stats_path = os.path.join(input_dir, stats_file_name)
            plot_path = os.path.join(input_dir, plot_file_name)
            plot_stats(stats_path, plot_path)
            print(f"Saved plot to {plot_path}")
    except subprocess.TimeoutExpired:
        wandb.log({"status": "timeout"})
        sys.exit(1)
    except Exception as e:
        wandb.log({"error": str(e)})
        sys.exit(1)
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument(
        "--python", type=str, default="/home/wzarzecki/miniforge3/envs/rf/bin/python"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="design")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--stats_file_name", default="stats.csv")
    parser.add_argument("--stride_path", type=str, default="./stride/stride")
    parser.add_argument("--plot_file_name", type=str, default="plot.png", help="Filename for output plot (saved in input_dir)")
    parser.add_argument("--mode", choices=["generation", "eval"], default="generation", help="Run mode: 'generation' (default) runs structure generation and optionally stride evaluation; 'eval' only runs stride evaluation and plotting.")
    args = parser.parse_args()
    print(f"running at {os.getcwd()}")
    print(f"running args: {args}")

    if args.mode == "generation":
        disable_structure_gen = False
        disable_stride_eval = True
    else:  # eval mode
        disable_structure_gen = True
        disable_stride_eval = False

    main(
        num_designs=args.num_designs,
        python=args.python,
        input_dir=args.input_dir,
        prefix=args.prefix,
        length=args.length,
        stats_file_name=args.stats_file_name,
        stride_path=args.stride_path,
        plot_file_name=args.plot_file_name,
        disable_structure_gen=disable_structure_gen,
        disable_stride_eval=disable_stride_eval,
    )
