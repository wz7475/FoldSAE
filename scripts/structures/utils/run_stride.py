from argparse import ArgumentParser
import os
import subprocess


def run_stride(pdb_path: str, output_path: str, stride_path):
    """Run STRIDE on a PDB file and save output to a file."""
    try:
        result = subprocess.run(
            f"{stride_path} -o {pdb_path} > {output_path}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Warning: STRIDE failed for {pdb_path}: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Error: STRIDE timed out for {pdb_path}")
        return False
    except Exception as e:
        print(f"Error running STRIDE on {pdb_path}: {e}")
        return False


def parse_directory(dir_with_pdb: str, dir_for_stride, stride_binary: str) -> None:
    """
    Recursively process all PDB files in the directory tree and create corresponding STRIDE files
    while maintaining the same directory structure.
    """
    os.makedirs(dir_for_stride, exist_ok=True)
    
    # First, count total PDB files for progress reporting
    total_pdb_files = 0
    for root, dirs, files in os.walk(dir_with_pdb):
        total_pdb_files += len([f for f in files if f.endswith(".pdb")])
    
    print(f"Found {total_pdb_files} PDB files to process")
    
    processed = 0
    successful = 0
    failed = 0
    
    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(dir_with_pdb):
        # Calculate relative path from the base PDB directory
        rel_path = os.path.relpath(root, dir_with_pdb)
        
        # Create corresponding directory in stride output
        if rel_path == ".":
            # This is the root directory
            stride_subdir = dir_for_stride
        else:
            # This is a subdirectory
            stride_subdir = os.path.join(dir_for_stride, rel_path)
            os.makedirs(stride_subdir, exist_ok=True)
        
        # Process all PDB files in this directory
        for pdb_file in files:
            if pdb_file.endswith(".pdb"):
                pdb_path = os.path.join(root, pdb_file)
                base_name = os.path.basename(pdb_file)
                stride_file_name = base_name.replace(".pdb", ".stride")
                stride_file = os.path.join(stride_subdir, stride_file_name)
                
                processed += 1
                print(f"Processing ({processed}/{total_pdb_files}): {pdb_path} -> {stride_file}")
                
                if run_stride(pdb_path, stride_file, stride_binary):
                    successful += 1
                else:
                    failed += 1
    
    print("\nSTRIDE processing completed:")
    print(f"  Total files: {total_pdb_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {dir_for_stride}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pdb_dir", default="/home/wzarzecki/ds_secondary_struct/structures")
    parser.add_argument("--stride_dir", default="/home/wzarzecki/ds_secondary_struct/stride")
    parser.add_argument("--stride_binary", default="/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride")
    args = parser.parse_args()
    parse_directory(args.pdb_dir, args.stride_dir, args.stride_binary)
