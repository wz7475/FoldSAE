import argparse
import glob

import pandas as pd

from ost import io
from ost.bindings import tmtools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_path',  help='Path to .pdb structured predicted by RFDiffusion')
    parser.add_argument('--af2_path',  help='Path to .pdb structured predicted by AF2')
    parser.add_argument('--output_path',  help='Output path')

    args = parser.parse_args()

    rf_files = sorted(glob.glob(f'{args.rf_path}/*.pdb'))
    af2_files = sorted(glob.glob(f'{args.af2_path}/*.pdb'))
    if len(rf_files) != len(af2_files):
        raise ValueError('Number of RF files and AF2 files are not equal.')
    ids = []
    rmsds = []
    tms = []
    for rf_file, af2_file in zip(rf_files, af2_files):
        rf_id = rf_file.split('/')[-1].strip('.pdb')
        af_id = af2_file.split('/')[-1].strip('.pdb')
        if rf_id not in af_id:
            print(f'Issue with {rf_file} and {af2_file}. Skipping evaluation.')
            continue
        pdb1 = io.LoadPDB(rf_file, restrict_chains='A')
        pdb2 = io.LoadPDB(af2_file, restrict_chains='A')
        result = tmtools.TMScore(pdb1, pdb2)
        ids.append(rf_id)
        rmsds.append(result.rmsd_common)
        tms.append(result.tm_score)

        pd.DataFrame.from_dict({
            'id': ids,
            'rmsd': rmsds,
            'tm': tms,
        }).to_csv(args.output_path, index=False)

