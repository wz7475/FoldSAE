import argparse

import seqme as sm

from val_utils import get_ref_seqs, read_sequences_from_dir


def validate_sequence(val_seqs: list[str], ref_seqs: list[str], embedder, name="val"):
    metrics = [
        sm.metrics.FBD(reference=ref_seqs, embedder=embedder),
        sm.metrics.MMD(reference=ref_seqs, embedder=embedder),
    ]

    return sm.evaluate({name: val_seqs}, metrics)


def main(val_seqs_dir, val_stride_dir, ref_seqs_dir, ref_stride_dir, n_ref, results_file):
    resampled_ref_seqs = get_ref_seqs(val_stride_dir, ref_seqs_dir, ref_stride_dir, n_ref)
    val_seqs = read_sequences_from_dir(None, val_seqs_dir)

    cache = sm.Cache(
        models={
            "esm2": sm.models.ESM2(
                model_name="facebook/esm2_t6_8M_UR50D", batch_size=256, device="cuda"
            )
        }
    )

    df = validate_sequence(val_seqs, resampled_ref_seqs, cache.model("esm2"))
    df.to_csv(results_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_seqs_dir", type=str, required=True, help="Directory for validation sequences")
    parser.add_argument("--val_stride_dir", type=str, required=True, help="Directory for validation stride files")
    parser.add_argument("--ref_seqs_dir", type=str, required=True, help="Directory for reference sequences")
    parser.add_argument("--ref_stride_dir", type=str, required=True, help="Directory for reference stride files")
    parser.add_argument("--n_ref", type=int, required=True, help="Number of reference sequences to resample")
    parser.add_argument("--results_file", type=str, required=True, help="Output file for results CSV")

    args = parser.parse_args()
    main(args.val_seqs_dir, args.val_stride_dir, args.ref_seqs_dir, args.ref_stride_dir, args.n_ref, args.results_file)
