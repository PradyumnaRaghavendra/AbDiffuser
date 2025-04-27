"""
Generate antibodies with the trained AbDiffuser model.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time

# Add the absolute path
sys.path.insert(0, '/content')

from abdiffuser.models.apmixer import APMixer
from abdiffuser.models.diffusion import GaussianDiffusion, DiscreteDiffusion, AbDiffuser
from abdiffuser.models.projection import ResidueProjection
from abdiffuser.models.priors import ResidueFrequencyPrior

def generate_antibodies(args):
    """Generate antibodies using a trained AbDiffuser model."""
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create model components
    apmixer = APMixer(
        num_layers=args.num_layers,
        dim=args.dim,
        use_side_chains=args.use_side_chains
    ).to(device)

    # Load residue prior if specified
    residue_prior = None
    if args.residue_prior:
        residue_prior = ResidueFrequencyPrior()
        residue_prior.load(args.residue_prior)
        print(f"Loaded residue frequency prior from {args.residue_prior}")

    # Create diffusion models
    atom_diffusion = GaussianDiffusion(
        noise_schedule='cosine',
        num_diffusion_steps=args.num_diffusion_steps,
    )

    aa_diffusion = DiscreteDiffusion(
        num_classes=21,
        num_diffusion_steps=args.num_diffusion_steps,
        position_specific_prior=residue_prior.get_all_distributions(device) if residue_prior else None
    )

    # Create projection layer
    projection_layer = ResidueProjection(use_side_chains=args.use_side_chains)

    # Create complete AbDiffuser model
    model = AbDiffuser(
        model=apmixer,
        atom_diffusion=atom_diffusion,
        aa_diffusion=aa_diffusion,
        projection_layer=projection_layer
    ).to(device)

    # Load trained model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from {args.checkpoint} (epoch {checkpoint['epoch']})")

    # Set model to eval mode
    model.eval()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate samples
    print(f"Generating {args.num_samples} antibody samples...")

    generated_samples = []

    with torch.no_grad():
        # Generate in batches
        num_batches = args.num_samples // args.batch_size
        if args.num_samples % args.batch_size > 0:
            num_batches += 1

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Determine batch size for this batch
            curr_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)

            # Generate samples
            x_pos, x_aa = model.sample(
                batch_size=curr_batch_size,
                num_residues=298,  # 2 × 149 for paired chains
                atom_types=4,      # N, CA, C, CB backbone atoms
                aa_types=21,       # 20 amino acids + gap
                device=device,
                temperature_pos=args.temperature_pos,
                temperature_aa=args.temperature_aa
            )

            # Convert to amino acid sequences
            sequences = []
            for i in range(curr_batch_size):
                # Get amino acid indices
                aa_indices = torch.argmax(x_aa[i], dim=1)

                # Convert to sequence
                aa_map = "ACDEFGHIKLMNPQRSTVWY-"  # Amino acid vocabulary
                seq = ''.join(aa_map[idx.item()] for idx in aa_indices)

                sequences.append(seq)

            # Create samples with positions and sequences
            for i in range(curr_batch_size):
                generated_samples.append({
                    'positions': x_pos[i].cpu().numpy(),
                    'one_hot': x_aa[i].cpu().numpy(),
                    'sequence': sequences[i]
                })

    # Save generated samples
    timestamp = int(time.time())
    output_path = os.path.join(args.output_dir, f"generated_samples_{timestamp}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(generated_samples, f)

    print(f"Saved {len(generated_samples)} generated samples to {output_path}")

    # Extract and save sequences
    seqs_path = os.path.join(args.output_dir, "generated_sequences.csv")
    seq_data = []

    for i, sample in enumerate(generated_samples):
        # Extract heavy chain (first half) and light chain (second half)
        sequence = sample['sequence']
        half_len = len(sequence) // 2
        heavy_chain = sequence[:half_len].replace('-', '')
        light_chain = sequence[half_len:].replace('-', '')

        seq_data.append({
            'sample_id': i,
            'heavy_chain': heavy_chain,
            'light_chain': light_chain,
            'full_sequence': sequence.replace('-', '')
        })

    seq_df = pd.DataFrame(seq_data)
    seq_df.to_csv(seqs_path, index=False)

    print(f"Saved sequences to {seqs_path}")

    # Generate and save sequence visualization
    plt.figure(figsize=(12, 6))

    # Visualize first sequence properties
    seq = seq_data[0]['full_sequence']

    # Plot amino acid distribution
    aa_counts = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        aa_counts[aa] = seq.count(aa)

    plt.bar(aa_counts.keys(), aa_counts.values())
    plt.title('Amino Acid Distribution in Generated Antibody')
    plt.xlabel('Amino Acid')
    plt.ylabel('Count')

    # Save plot
    plt.savefig(os.path.join(args.output_dir, "aa_distribution.png"))
    plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate antibodies with AbDiffuser")

    # Model options
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to trained model checkpoint")
    parser.add_argument("--num_layers", type=int, default=3,  # Reduced for efficiency
                      help="Number of APMixer layers")
    parser.add_argument("--dim", type=int, default=320,       # Reduced for efficiency
                      help="Hidden dimension")
    parser.add_argument("--use_side_chains", action="store_true",
                      help="Whether to model side chains")
    parser.add_argument("--num_diffusion_steps", type=int, default=200,  # Reduced for efficiency
                      help="Number of diffusion steps")

    # Prior options
    parser.add_argument("--residue_prior", type=str, default=None,
                      help="Path to residue frequency prior")

    # Generation options
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size for generation")
    parser.add_argument("--temperature_pos", type=float, default=1.0,
                      help="Temperature for position sampling")
    parser.add_argument("--temperature_aa", type=float, default=0.8,
                      help="Temperature for amino acid sampling")
    parser.add_argument("--output_dir", type=str, default="experiments/outputs",
                      help="Directory to save generated samples")

    # Hardware options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Generate antibodies
    generate_antibodies(args)

if __name__ == "__main__":
    main()
