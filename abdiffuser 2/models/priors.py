"""
Informative diffusion priors for AbDiffuser.
"""

import torch
import os

class ResidueFrequencyPrior:
    """Position-specific residue frequency prior for antibody sequences."""

    def __init__(self,
                num_positions=298,  # 2 × 149 for paired antibody chains
                num_aa_types=21):   # 20 amino acids + gap
        """Initialize the residue frequency prior."""
        self.num_positions = num_positions
        self.num_aa_types = num_aa_types

        # Initialize uniform distributions
        self.frequencies = torch.ones((num_positions, num_aa_types)) / num_aa_types

    def compute_from_data(self, sequences):
        """Compute position-specific frequencies from sequence data."""
        # Count occurrences of each amino acid at each position
        counts = sequences.sum(dim=0)

        # Add smoothing
        counts = counts + 0.01

        # Normalize to get frequencies
        self.frequencies = counts / torch.sum(counts, dim=1, keepdim=True)

    def save(self, path):
        """Save the frequencies to a file."""
        torch.save(self.frequencies, path)

    def load(self, path):
        """Load frequencies from a file."""
        if os.path.exists(path):
            self.frequencies = torch.load(path)
            self.num_positions, self.num_aa_types = self.frequencies.shape
        else:
            print(f"Warning: Prior file {path} not found, using uniform distribution")

    def get_all_distributions(self, device=torch.device('cpu')):
        """Get all residue frequency distributions."""
        return self.frequencies.to(device)
