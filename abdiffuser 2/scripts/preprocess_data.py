
"""
Modified preprocess script to handle OAS dataset format correctly.
"""

import os
import sys
import argparse
import gzip
import pickle
import json
import pandas as pd
import numpy as np
import torch
from tqdm.notebook import tqdm

# Add the absolute path
sys.path.insert(0, '/content')

from abdiffuser.utils.aho_numbering import AhoNumbering
from abdiffuser.models.priors import ResidueFrequencyPrior

def preprocess_oas_dataset(args):
    """Preprocess OAS dataset."""
    print(f"Preprocessing OAS dataset from {args.oas_dir}...")
    
    # Find all data files
    files = []
    for root, _, filenames in os.walk(args.oas_dir):
        for filename in filenames:
            if filename.endswith('.csv.gz'):
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} OAS data files")
    
    # If max_files is specified, limit the number of files
    if args.max_files > 0:
        files = files[:args.max_files]
        print(f"Using {len(files)} files for processing")
    
    # Process each file
    all_sequences = []
    aho = AhoNumbering()
    
    for file_path in tqdm(files, desc="Processing OAS files"):
        try:
            # Read CSV file - SKIP FIRST ROW which contains metadata
            with gzip.open(file_path, 'rt') as f:
                # Skip the first line (metadata)
                next(f)
                # Read the rest as CSV
                df = pd.read_csv(f)
            
            # Extract sequences
            sequences = extract_sequences_from_df(df, file_path, aho, args.max_sequences_per_file)
            all_sequences.extend(sequences)
            
            # Check if we have enough sequences
            if args.max_sequences > 0 and len(all_sequences) >= args.max_sequences:
                all_sequences = all_sequences[:args.max_sequences]
                break
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"Processed {len(all_sequences)} sequences total")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save processed data
    with open(args.output, 'wb') as f:
        pickle.dump(all_sequences, f)
    
    print(f"Saved processed data to {args.output}")
    
    # Generate residue frequency prior
    if args.generate_priors and all_sequences:
        generate_residue_frequency_prior(all_sequences, os.path.dirname(args.output))

# Keep the rest of the file the same...
def extract_sequences_from_df(df, file_path, aho, max_sequences=None):
    """Extract sequences from a pandas DataFrame."""
    sequences = []

    # Find columns containing aligned amino acid sequences
    heavy_cols = [col for col in df.columns if 'heavy' in col.lower() and 'aa' in col.lower()]
    light_cols = [col for col in df.columns if 'light' in col.lower() and 'aa' in col.lower()]

    # Use sequence_alignment_aa_heavy/light if available
    heavy_col = next((col for col in heavy_cols if 'sequence_alignment_aa_heavy' in col),
                     heavy_cols[0] if heavy_cols else None)
    light_col = next((col for col in light_cols if 'sequence_alignment_aa_light' in col),
                     light_cols[0] if light_cols else None)

    if heavy_col is None or light_col is None:
        print(f"Could not find heavy/light chain columns in {file_path}")
        return sequences

    # Process each row
    for idx, row in df.iterrows():
        if max_sequences and len(sequences) >= max_sequences:
            break

        try:
            heavy_seq = str(row[heavy_col]).upper()
            light_seq = str(row[light_col]).upper()

            # Skip if sequences are too short
            if len(heavy_seq) < 10 or len(light_seq) < 10:
                continue

            # Clean sequences
            heavy_seq = ''.join(aa for aa in heavy_seq if aa in 'ACDEFGHIKLMNPQRSTVWY-')
            light_seq = ''.join(aa for aa in light_seq if aa in 'ACDEFGHIKLMNPQRSTVWY-')

            # Apply AHo numbering
            try:
                one_hot, non_gap_positions = aho.number_sequence(heavy_seq, light_seq)
                sequence = {
                    'heavy_seq': heavy_seq,
                    'light_seq': light_seq,
                    'one_hot': one_hot,
                    'non_gap_positions': non_gap_positions
                }
                sequences.append(sequence)
            except Exception as e:
                continue

        except Exception as e:
            continue

    return sequences

def preprocess_her2_dataset(args):
    """Preprocess HER2 binder dataset."""
    print(f"Preprocessing HER2 dataset...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load train, val, test splits
    train_df = pd.read_csv(os.path.join(args.her2_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(args.her2_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(args.her2_dir, 'test.csv'))

    print(f"Loaded HER2 dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    # Original Trastuzumab heavy chain sequence
    trastuzumab_heavy = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"

    # CDR H3 region position in the sequence (Chothia numbering)
    cdr_h3_start = 95

    # Process each split
    process_her2_split(train_df, 'train', args.output_dir, trastuzumab_heavy, cdr_h3_start)
    process_her2_split(val_df, 'val', args.output_dir, trastuzumab_heavy, cdr_h3_start)
    process_her2_split(test_df, 'test', args.output_dir, trastuzumab_heavy, cdr_h3_start)

    print(f"Processed HER2 dataset saved to {args.output_dir}")

def process_her2_split(df, split_name, output_dir, trastuzumab_heavy, cdr_h3_start):
    """Process a split of the HER2 binder dataset."""
    # Sequence column name may vary
    sequence_col = 'seq' if 'seq' in df.columns else 'sequence'
    label_col = 'label' if 'label' in df.columns else 'is_binder'

    # Initialize AHo numbering
    aho = AhoNumbering()

    # Process each sequence
    processed_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        try:
            # Get CDR H3 sequence and binding information
            cdr_h3_seq = row[sequence_col]
            is_binder = int(row[label_col])

            # Create full heavy chain by replacing the CDR H3 region
            full_seq = list(trastuzumab_heavy)
            for i, aa in enumerate(cdr_h3_seq):
                pos = cdr_h3_start + i
                if pos < len(full_seq):
                    full_seq[pos] = aa

            full_seq = ''.join(full_seq)

            # Apply AHo numbering
            # For HER2, we don't have the light chain, so use a placeholder
            light_chain_placeholder = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT"

            one_hot, non_gap_positions = aho.number_sequence(full_seq, light_chain_placeholder)

            processed_data.append({
                'heavy_seq': full_seq,
                'light_seq': light_chain_placeholder,
                'cdr_h3_seq': cdr_h3_seq,
                'one_hot': one_hot,
                'non_gap_positions': non_gap_positions,
                'is_binder': is_binder,
            })

        except Exception as e:
            continue

    # Save processed data
    output_path = os.path.join(output_dir, f"{split_name}_processed.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Saved {len(processed_data)} processed sequences to {output_path}")

def generate_residue_frequency_prior(sequences, output_dir):
    """Generate position-specific residue frequency prior."""
    print("Generating residue frequency prior...")

    # Create prior
    prior = ResidueFrequencyPrior()

    # Extract one-hot sequences
    one_hot_sequences = []
    for seq in sequences:
        if 'one_hot' in seq:
            one_hot_sequences.append(torch.tensor(seq['one_hot']))

    # Stack sequences
    if one_hot_sequences:
        stacked_sequences = torch.stack(one_hot_sequences, dim=0)

        # Compute frequencies
        prior.compute_from_data(stacked_sequences)

        # Save prior
        output_path = os.path.join(output_dir, "residue_frequency_prior.pt")
        prior.save(output_path)

        print(f"Saved residue frequency prior to {output_path}")
    else:
        print("No one-hot sequences found, skipping residue frequency prior generation")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Preprocess data for AbDiffuser")

    # General options
    parser.add_argument("--mode", type=str, default="oas", choices=["oas", "her2", "both"],
                      help="Dataset to preprocess (oas, her2, or both)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                      help="Directory to save processed data")

    # OAS options
    parser.add_argument("--oas_dir", type=str, default="data/OAS",
                      help="Directory containing OAS data files")
    parser.add_argument("--output", type=str, default="data/processed/oas_aligned.pkl",
                      help="Output file for processed OAS data")
    parser.add_argument("--max_files", type=int, default=1,
                      help="Maximum number of files to process (0 for all)")
    parser.add_argument("--max_sequences", type=int, default=1000,
                      help="Maximum number of sequences to process (0 for all)")
    parser.add_argument("--max_sequences_per_file", type=int, default=1000,
                      help="Maximum number of sequences to extract from each file")

    # HER2 options
    parser.add_argument("--her2_dir", type=str, default="data/HER2",
                      help="Directory containing HER2 data files")

    # Prior options
    parser.add_argument("--generate_priors", action="store_true",
                      help="Generate residue frequency priors")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process datasets
    if args.mode in ["oas", "both"]:
        preprocess_oas_dataset(args)

    if args.mode in ["her2", "both"]:
        preprocess_her2_dataset(args)

if __name__ == "__main__":
    main()