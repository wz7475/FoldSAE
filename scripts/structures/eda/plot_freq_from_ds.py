import argparse
from collections import Counter
from datasets import Dataset
import matplotlib.pyplot as plt

def main(input_dir, output_img):
    ds = Dataset.load_from_disk(input_dir)
    all_annotations = ds["secondary_struct"]
    counter = Counter(all_annotations)
    keys = [str(x) for x in counter.keys()]
    vals = list(counter.values())
    total = sum(vals)
    vals = [x / total for x in vals]

    ghi_count = counter.get("G", 0) + counter.get("H", 0) + counter.get("I", 0)
    eb_count = counter.get("E", 0) + counter.get("B", 0) + counter.get("b", 0)
    none_count = counter.get(None, 0)
    others_count = total - ghi_count - eb_count - none_count
    grouped_keys = ["G,H,I", "E,B,b", "None", "others"]
    grouped_vals = [x / total for x in [ghi_count, eb_count, none_count, others_count]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(keys, vals)
    ax[0].set_title("Each kind independent", fontsize=10)
    ax[1].bar(grouped_keys, grouped_vals)
    ax[1].set_title("GHI - alpha helix; EB - beta sheet", fontsize=10)
    fig.suptitle("Ratio of secondary structures kinds in probes training dataset")
    plt.tight_layout()
    plt.savefig(output_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot secondary structure frequencies.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input dataset directory")
    parser.add_argument("--output_img", type=str, required=True, help="Path to save output image")
    args = parser.parse_args()
    main(args.input_dir, args.output_img)
