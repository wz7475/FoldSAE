#!/usr/bin/env python3
"""
Copy selected design PDBs (and optionally STRIDE files) for all lambdas.

This script mirrors the directory scanning pattern used by lambda_x_seed_table.py:
it looks for directories named: lambda_{value}_thr_{threshold}_{class}/ under --base_dir.

Given a list of protein IDs (design seed basenames without extension), it will copy
any matching `{id_prefix}{id}.pdb` found in those directories into --target_dir.

Default behavior mirrors the relative source directory structure. When --subdir_suffix
is provided (defaults to "0.15._beta"), the script only copies from directories whose
names end with that suffix and places PDBs into flat per-lambda subdirectories under
--target_dir (e.g., target/lambda_0.15/...). In this mode, only PDBs are copied.
"""

import os
import argparse
import re
import shutil
from typing import Iterable, List, Set, Tuple


def parse_ids(ids_args: Iterable[str]) -> List[str]:
    """Normalize ids passed as multiple args and/or comma-separated values."""
    result: List[str] = []
    for token in ids_args:
        for part in token.split(","):
            part = part.strip()
            if part:
                result.append(part)
    # de-duplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def matches_lambda_dir(name: str) -> Tuple[bool, Tuple[str, str, str]]:
    """Return whether the directory name matches the lambda sweep pattern and capture groups."""
    m = re.match(r"^lambda_([^_]+)_thr_([^_]+)_([^_]+)$", name)
    if not m:
        return False, ("", "", "")
    return True, (m.group(1), m.group(2), m.group(3))


def copy_if_exists(src_file: str, dst_file: str, overwrite: bool) -> bool:
    if not os.path.exists(src_file):
        return False
    if os.path.exists(dst_file) and not overwrite:
        return False
    ensure_parent_dir(dst_file)
    shutil.copy2(src_file, dst_file)
    return True


def scan_and_copy(base_dir: str, target_dir: str, ids: List[str], id_prefix: str, also_stride: bool, overwrite: bool, subdir_suffix: str) -> None:
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    # If overwrite is enabled, remove the entire target directory to ensure a clean copy
    if overwrite and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    total_found = 0
    total_copied = 0
    per_id_found = {pid: 0 for pid in ids}
    per_id_copied = {pid: 0 for pid in ids}

    for root, dirs, files in os.walk(base_dir):
        # Filter to immediate lambda_* directories; but allow deeper walking
        for dir_name in list(dirs):
            is_match, parts = matches_lambda_dir(dir_name)
            if not is_match:
                continue
            # If suffix filtering is enabled, only process directories ending with the suffix
            if subdir_suffix:
                if not dir_name.endswith(subdir_suffix):
                    continue

            lambda_dir = os.path.join(root, dir_name)
            for pid in ids:
                pdb_name = f"{id_prefix}{pid}.pdb"
                pdb_src = os.path.join(lambda_dir, pdb_name)
                if os.path.exists(pdb_src):
                    per_id_found[pid] += 1
                    total_found += 1

                    # Flat per-lambda target if suffix filtering is used; otherwise mirror structure
                    if subdir_suffix:
                        lambda_value = parts[0]
                        flat_dir = os.path.join(target_dir, f"lambda_{lambda_value}")
                        pdb_dst = os.path.join(flat_dir, os.path.basename(pdb_src))
                    else:
                        rel_path = os.path.relpath(pdb_src, start=base_dir)
                        pdb_dst = os.path.join(target_dir, rel_path)
                    did_copy = copy_if_exists(pdb_src, pdb_dst, overwrite)
                    if did_copy:
                        per_id_copied[pid] += 1
                        total_copied += 1

                    # Only copy STRIDE files in mirroring mode (no suffix filter)
                    if also_stride and not subdir_suffix:
                        stride_src = pdb_src[:-4] + ".stride"
                        rel_stride = os.path.relpath(stride_src, start=base_dir)
                        stride_dst = os.path.join(target_dir, rel_stride)
                        copy_if_exists(stride_src, stride_dst, overwrite)

    print(f"Found {total_found} matching PDBs across all lambdas. Copied {total_copied}.")
    for pid in ids:
        print(f"  ID={pid}: found={per_id_found[pid]}, copied={per_id_copied[pid]}")


def main():
    parser = argparse.ArgumentParser(description="Copy selected design PDBs for all lambdas, preserving structure")
    parser.add_argument("--base_dir", required=True, help="Root directory containing lambda_*_thr_*_* subdirectories")
    parser.add_argument("--target_dir", required=True, help="Destination root directory to mirror copies into")
    parser.add_argument("--ids", nargs="+", required=True, help="Protein IDs (seeds) to copy; space and/or comma separated")
    parser.add_argument("--id_prefix", default="design_", help="Filename prefix before each ID, e.g., 'design_' -> design_<ID>.pdb")
    parser.add_argument("--also_stride", action="store_true", help="Also copy .stride files alongside PDBs if present (ignored with --subdir_suffix)")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing files at the destination; when true, target dir is removed first")
    parser.add_argument("--subdir_suffix", default="0.15_beta", help="Only copy from source subdirectories ending with this suffix; copies PDBs into flat per-lambda subdirs")

    args = parser.parse_args()

    ids = parse_ids(args.ids)
    if not ids:
        raise SystemExit("No protein IDs provided after parsing.")

    print(f"Scanning base directory: {args.base_dir}")
    print(f"Target directory: {args.target_dir}")
    print(f"IDs: {', '.join(ids)}")
    print(f"ID prefix: {args.id_prefix}")
    if args.also_stride:
        print("Also copying .stride files")
    if args.overwrite:
        print("Overwrite mode enabled")
    if args.subdir_suffix:
        print(f"Filtering source directories by suffix: '{args.subdir_suffix}' (flat per-lambda output)")

    scan_and_copy(
        base_dir=args.base_dir,
        # target_dir=os.path.join(args.target_dir, "pdb"),
        target_dir=args.target_dir,
        ids=ids,
        id_prefix=args.id_prefix,
        also_stride=args.also_stride,
        overwrite=args.overwrite,
        subdir_suffix=args.subdir_suffix,
    )


if __name__ == "__main__":
    main()


