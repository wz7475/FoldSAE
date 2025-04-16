import glob
import argparse
from enum import Enum
import pandas as pd
from matplotlib import pyplot as plt


class Membrane(Enum):
    MEMBRANE = "Membrane bound"
    SOLUBLE = "Soluble"
    # UNKNOWN = "?"

    def __str__(self):
        return {
            self.MEMBRANE: "Membrane bound",
            self.SOLUBLE: "Soluble",
            # self.UNKNOWN: "Unknown",
        }.get(self)



class Location(Enum):
    CELL_MEMBRANE = "Cell-Membrane"
    CYTOPLASM = "Cytoplasm"
    ENDOPLASMATIC_RETICULUM = "Endoplasmic reticulum"
    GOLGI_APPARATUS = "Golgi - Apparatus"
    LYSOSOME_OR_VACUOLE = "Lysosome / Vacuole"
    MITOCHONDRION = "Mitochondrion"
    NUCLEUS = "Nucleus"
    PEROXISOME = "Peroxisome"
    PLASTID = "Plastid"
    EXTRACELLULAR = "Extra - cellular"
    # UNKNOWN = "?"

    def __str__(self):
        return {
            self.CELL_MEMBRANE: "Cell-Membrane",
            self.CYTOPLASM: "Cytoplasm",
            self.ENDOPLASMATIC_RETICULUM: "Endoplasmic reticulum",
            self.GOLGI_APPARATUS: "Golgi - Apparatus",
            self.LYSOSOME_OR_VACUOLE: "Lysosome / Vacuole",
            self.MITOCHONDRION: "Mitochondrion",
            self.NUCLEUS: "Nucleus",
            self.PEROXISOME: "Peroxisome",
            self.PLASTID: "Plastid",
            self.EXTRACELLULAR: "Extra - cellular",
            # self.UNKNOWN: "?",
        }.get(self)

LOCATION_VALUES = [e.value for e  in Location]
MEMBRANE_VALUES = [e.value for e  in Membrane]

LOCATION_COLUMN = "Subcellular Localization"
MEMBRANE_COLUMN = "Solubility/Membrane-boundness"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_with_csvs", default="ablation_test")
    parser.add_argument("--filename", default="classifiers_bio_emb.csv")
    parser.add_argument("--subcellular_path", default="subcellular.png")
    parser.add_argument("--solubility_path", default="solubility.png")
    args = parser.parse_args()
    input_dir = args.dir_with_csvs
    filename = args.filename

    # collect list of dicts from csvs
    list_of_dicts = []
    for name in glob.glob(f"{input_dir}/**/{filename}", recursive=True):
        print(name)
        block_name = name.split("/")[-2]
        df = pd.read_csv(name)
        list_of_dicts.append({
            "name": block_name,
            LOCATION_COLUMN: df[LOCATION_COLUMN].value_counts(),
            MEMBRANE_COLUMN: df[MEMBRANE_COLUMN].value_counts()
        })

    number_of_all_seqs = len(list_of_dicts) * len(df)
    # merge into single csv
    values_columns = LOCATION_VALUES + MEMBRANE_VALUES
    values_lists = [[values_dict[LOCATION_COLUMN].get(key, 0) for values_dict in list_of_dicts] for key in LOCATION_VALUES]
    values_lists += [[values_dict[MEMBRANE_COLUMN].get(key, 0) for values_dict in list_of_dicts] for key in MEMBRANE_VALUES]
    values_lists += [[values_dict["name"] for values_dict in list_of_dicts]]
    final_df = pd.DataFrame(list(zip(*values_lists)), columns=values_columns + ["name"]).sort_values(by="name").set_index("name")

    # for column in values_columns:
    #     final_df[column] /= number_of_all_seqs
    y_max = final_df[LOCATION_VALUES].max().max()


    # for columns in target_columns:
    axes = final_df[LOCATION_VALUES].plot.bar(subplots=True, figsize=(8, 16))
    for ax in axes:
        ax.set_ylim(0, 1.1 * y_max)
    plt.subplots_adjust(top=0.98, bottom=.1)
    plt.savefig(args.subcellular_path)


    y_max = final_df[MEMBRANE_VALUES].max().max()
    final_df.to_csv("rf_diffusion_ablations.csv")

    # for columns in target_columns:
    axes = final_df[MEMBRANE_VALUES].plot.bar(subplots=True, figsize=(8, 4))
    for ax in axes:
        ax.set_ylim(0, 1.1 * y_max)
    # plt.subplots_adjust(top=0.98, bottom=.02)
    plt.subplots_adjust(top=0.9, bottom=.3)
    plt.savefig(args.solubility_path)