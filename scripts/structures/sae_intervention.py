import os
import subprocess
import time
from argparse import ArgumentParser

import wandb
import sys

import pandas as pd
import glob
import matplotlib.pyplot as plt


def extract_indices_filename(indices_path: str) -> str:
    """Extract filename from indices path for subdirectory naming."""
    return os.path.splitext(os.path.basename(indices_path))[0]


def generate_config_file(
    lambda_val: float, indices_path: str, output_config_name: str, python: str
) -> str:
    """Run generate_config_structures.py to create config file."""
    command = f"""
{python} RFDiffSAE/scripts/generate_config_structures.py \
  --lambda_ {lambda_val} \
  --indices_path {indices_path} \
  --output_config_name {output_config_name}
"""
    print(f"running command: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate config: {result.stderr}")




def prepare_command_rfdiffusion(
    input_dir,
    prefix,
    length,
    num_designs,
    config_name,
    python="/home/wzarzecki/miniforge3/envs/rf/bin/python",
) -> str:
    return f"""
{python} RFDiffSAE/scripts/run_inference.py \
  inference.output_prefix={input_dir}/{prefix} \
  'contigmap.contigs=[{length}-{length}]' \
  inference.num_designs={num_designs} \
  inference.final_step=1 \
  inference.use_random_suffix_for_new_design=False \
  inference.seed=1 \
  saeinterventions={config_name}
"""


def run_rfdiffusion(command):
    print(f"running command: {command}")
    tic = time.time()
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3600 * 48,  # Set appropriate timeout
    )
    print(f"elapsed time running RFdiffusion: {time.time() - tic}")
    return result


def log_result(wandb_handle, result):
    wandb_handle.log(
        {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def eval_structures(
    input_dir: str, stats_file_name: str, stride_path: str, plot_file_name: str
):
    """Evaluate pdbs under input_dir by running stride into a mirrored stride directory,
    then compute per-first-level-directory / per-lambda stats and plots.

    input_dir layout assumed: input_dir/<first_level>/<lambda_val>/*.pdb
    Creates: input_dir/_stride/<first_level>/<lambda_val>/*.stride
    For each first_level directory creates a CSV (stats_file_name) and a plot (plot_file_name
    with first_level name prefixed) saved into the first_level directory.
    """
    # helpers
    def extract_helix_counts_from_stride(stride_path_file: str):
        # count occurrences of G/H/I in STR lines as helix indicator and total other structure
        ghi = 0
        other = 0
        if not os.path.exists(stride_path_file):
            return 0, 0
        with open(stride_path_file) as f:
            for line in f:
                if line.startswith("STR"):
                    ghi += line.count("G")
                    ghi += line.count("H")
                    ghi += line.count("I")
                if line.startswith("SEQ"):
                    # count glycine residues in SEQ lines as "other" per request
                    other += line.count("G")
        return ghi, other

    def run_stride_on_pdb(pdb_path: str, out_stride_path: str):
        os.makedirs(os.path.dirname(out_stride_path), exist_ok=True)
        cmd = f"{stride_path} -o {pdb_path} > {out_stride_path}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

    # create mirrored stride base
    stride_base = os.path.join(input_dir, "_stride")
    os.makedirs(stride_base, exist_ok=True)

    # Walk first-level directories under input_dir

    import re
    for first in sorted(os.listdir(input_dir)):
        first_path = os.path.join(input_dir, first)
        # skip the stride mirror dir
        if not os.path.isdir(first_path) or first == "_stride":
            continue

        # Extract number of coefficients from directory name (e.g., baseline_top_100 -> 100)
        match = re.search(r"(\d+)", first)
        num_coeffs = match.group(1) if match else "?"

        stats_rows = []

        # Each first-level dir contains lambda subdirs; sort them numerically when possible
        lambda_dirs = [d for d in os.listdir(first_path) if os.path.isdir(os.path.join(first_path, d))]
        def _numeric_key(s):
            try:
                return float(s)
            except Exception:
                return float('inf')

        for lambda_name in sorted(lambda_dirs, key=_numeric_key):
            lambda_path = os.path.join(first_path, lambda_name)

            # mirrored stride dir for this lambda
            mirrored_stride_dir = os.path.join(stride_base, first, lambda_name)
            os.makedirs(mirrored_stride_dir, exist_ok=True)

            pdb_count = 0
            total_helix = 0
            total_other = 0

            # find pdb files directly under lambda_path (non-recursive)
            for entry in sorted(os.listdir(lambda_path)):
                if not entry.endswith(".pdb"):
                    continue
                pdb_count += 1
                pdb_file = os.path.join(lambda_path, entry)
                stride_file = os.path.join(mirrored_stride_dir, entry.replace(".pdb", ".stride"))
                try:
                    run_stride_on_pdb(pdb_file, stride_file)
                except subprocess.TimeoutExpired:
                    wandb.log({"status": "stride_timeout", "file": pdb_file})
                    continue
                helix, other = extract_helix_counts_from_stride(stride_file)
                total_helix += helix
                total_other += other

            helix_ratio = (total_helix / (total_helix + total_other)) if (total_helix + total_other) > 0 else 0.0

            stats_rows.append(
                {
                    "first_level": first,
                    "lambda": lambda_name,
                    "pdb_count": pdb_count,
                    "helix_count": total_helix,
                    "other_count": total_other,
                    "helix_ratio": helix_ratio,
                }
            )

        # write CSV for this first-level dir
        if stats_rows:
            df = pd.DataFrame(stats_rows)
            # ensure numeric sort order for lambda when plotting
            try:
                df['lambda_num'] = pd.to_numeric(df['lambda'], errors='coerce')
                df = df.sort_values('lambda_num', ascending=True)
            except Exception:
                pass
            stats_csv_path = os.path.join(first_path, stats_file_name)
            df.to_csv(stats_csv_path, index=False)
            print(f"savig stats to {stats_csv_path}")

            # create bar plot: for each lambda show two bars: pdb_count and helix_ratio (scaled)
            # To show both on same plot, we'll use twin axis: left axis for pdb_count, right axis for helix_ratio
            fig, ax1 = plt.subplots(figsize=(8, 5))
            lambdas = df['lambda'].astype(str).tolist()
            x = list(range(len(lambdas)))

            ax1.bar(x, df['pdb_count'].tolist(), color='C0', alpha=0.7, label='pdb_count')
            ax1.set_xlabel('lambda')
            ax1.set_ylabel('pdb count', color='C0')
            ax1.set_xticks(x)
            ax1.set_xticklabels(lambdas, rotation=45)

            # Add plot title with number of coefficients
            ax1.set_title(f"{first} ({num_coeffs} coefficients)")

            ax2 = ax1.twinx()
            # plot helix_ratio as a bar on the secondary axis (scaled 0..1)
            width = 0.35
            ax2.bar([i + width for i in x], df['helix_ratio'], width, color='C1', alpha=0.7, label='helix_ratio')
            ax2.set_ylabel('helix ratio (helix / total)', color='C1')

            # legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

            plot_path = os.path.join(first_path, f"{first}_{plot_file_name}")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"saving image to  {plot_path}")
            wandb.log({f"plot_{first}": wandb.Image(plot_path)})


def main(
    num_designs,
    python,
    input_dir,
    prefix,
    lambda_val,
    indices_path,
    stats_file_name,
    stride_path,
    plot_file_name,
    disable_structure_gen=False,
    disable_stride_eval=True,
):
    wandb.init()

    try:
        # Only perform generation-only setup when structure generation is enabled
        if not disable_structure_gen:
            # Extract indices filename for subdirectory naming
            indices_filename = extract_indices_filename(indices_path)

            # Create subdirectory structure: input_dir/indices_filename/lambda/
            lambda_subdir = f"{input_dir}/{indices_filename}/{lambda_val}"
            os.makedirs(lambda_subdir, exist_ok=True)

            # Generate unique config name based on lambda and indices
            config_name = f"lambda_{lambda_val}_{indices_filename}.yaml"

            ########################################
            ##### 1) generate config file
            print(f"Generating config file: {config_name}")
            generate_config_file(
                lambda_val, indices_path, config_name, python
            )
            print(f"Generated config at: {config_name}")

            ########################################
            ##### 2) generate structures with RFDiffusion
            command = prepare_command_rfdiffusion(
                lambda_subdir,
                prefix,
                150,
                num_designs,
                config_name,
                python,  # Fixed length=150
            )
            result = run_rfdiffusion(command)
            print("executed rfdiffusion")
            if result.returncode != 0:
                sys.exit(result.returncode)
            log_result(wandb, result)
            print("logged results")

        ########################################
        ##### 3) parse results with Stride parser (if in eval mode)
        if not disable_stride_eval:
            # In eval mode we operate on the provided input_dir directly
            eval_structures(input_dir, stats_file_name, stride_path, plot_file_name)

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
    parser.add_argument(
        "--lambda_val",
        type=float,
        default=0.0,
        help="Lambda value for SAE intervention",
    )
    parser.add_argument(
        "--indices_path", type=str, default="indices.pt", help="Path to indices file"
    )
    parser.add_argument("--stats_file_name", type=str, default="stats.csv")
    parser.add_argument("--stride_path", type=str, default="./stride/stride")
    parser.add_argument(
        "--plot_file_name",
        type=str,
        default="plot.png",
        help="Filename for output plot (saved in input_dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["generation", "eval"],
        default="generation",
        help="Run mode: 'generation' (default) runs structure generation and optionally stride evaluation; 'eval' only runs stride evaluation and plotting.",
    )
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
        lambda_val=args.lambda_val,
        indices_path=args.indices_path,
        stats_file_name=args.stats_file_name,
        stride_path=args.stride_path,
        plot_file_name=args.plot_file_name,
        disable_structure_gen=disable_structure_gen,
        disable_stride_eval=disable_stride_eval,
    )
