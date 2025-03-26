import os
from argparse import ArgumentParser
from typing import Tuple

import pandas as pd
import pycdhit
from matplotlib import pyplot as plt

TEMP_FASTA_NAME = "temp.fasta"
TEMP_CDHIT_FASTA_NAME = "temp_cdhit.fasta"
thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_directory(path: str, fasta_ext) -> Tuple[int, list[float]]:
    """
    input: director path
    output: block_num (int), list of ratios original / after cdhit with thresholds [0.7, 0.8, 0.9] (list[float)
    """
    block_num = int(path.split("_")[-1])
    os.system(f'find {path} -name "*.{fasta_ext}" -exec cat {{}} \; > {TEMP_FASTA_NAME}')
    ratios = []
    for thr in thresholds:
        pycdhit.cd_hit(
            i=TEMP_FASTA_NAME,
            o=TEMP_CDHIT_FASTA_NAME,
            c=thr,
            n=3,
            d=0,
            sc=1,
        )
        with open(TEMP_CDHIT_FASTA_NAME) as fp:
            output_lines = fp.readlines()
        with open(TEMP_FASTA_NAME) as fp:
            input_lines = fp.readlines()
        ratios.append(len(output_lines) / len(input_lines))
    return block_num, ratios


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--root", help="root for directories from input")
    parser.add_argument("-e", "--ext", default="fa", help="extention of fasta file")
    parser.add_argument("--output", "-o", default="sequence_diversity.png")
    args = parser.parse_args()
    ext = args.ext
    root = args.root
    blocks = []
    all_ratios = []
    for dir in sorted(os.listdir(root)):
        path = os.path.join(root, dir)
        block, ratios = parse_directory(path, ext)
        blocks.append(block)
        all_ratios.append(ratios)
        print(dir)
    df = pd.DataFrame([(block, *ration) for block, ration in zip(blocks, all_ratios)],
                      columns=["block"] + [str(t) for t in thresholds])
    df.to_csv("div.csv", index=False)
    df = pd.read_csv("div.csv")
    df = df.set_index("block")
    df = df.sort_index()
    df.plot.bar(figsize=(20, 8), width=0.8)
    plt.legend(loc='lower left')
    plt.subplots_adjust(left=0.05, right=0.98)
    plt.ylabel("diversity rate")
    plt.savefig("membrane.png")
