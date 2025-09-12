from argparse import ArgumentParser
import glob
import os
import subprocess


def run_stride(pdb_path: str, output_path: str, stride_path):
    subprocess.run(
        f"{stride_path} -o {pdb_path} > {output_path}",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def parse_directory(dir_with_pdb: str, dir_for_stride, stride_binary: str) -> None:
    os.makedirs(dir_for_stride, exist_ok=True)
    for pdb_file in glob.glob(f"{dir_with_pdb}/*.pdb"):
        base_name = os.path.basename(pdb_file)
        stride_file_name = base_name.replace(".pdb", ".stride")
        stride_file = os.path.join(dir_for_stride, stride_file_name)
        run_stride(pdb_file, stride_file, stride_binary)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pdb_dir", default="/home/wzarzecki/ds_secondary_struct/structures")
    parser.add_argument("--stride_dir", default="/home/wzarzecki/ds_secondary_struct/stride")
    parser.add_argument("--stride_binary", default="/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride")
    args = parser.parse_args()
    parse_directory(args.pdb_dir, args.stride_dir, args.stride_binary)
