"""
AHo numbering utilities for antibody sequences.
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

class AhoNumbering:
    """
    Class for converting antibody sequences to AHo numbering scheme.

    The AHo numbering scheme represents antibody sequences in a fixed-length format.
    """

    def __init__(self):
        """Initialize the AHo numbering class."""
        # Fixed positions in AHo numbering (H and L chains both have 149 positions)
        self.num_positions_per_chain = 149
        self.total_positions = 2 * self.num_positions_per_chain  # H + L

    def number_sequence(self,
                        heavy_seq: str,
                        light_seq: str) -> Tuple[np.ndarray, List[int]]:
        """Apply AHo numbering to antibody heavy and light chain sequences."""
        # Use simplified numbering for Colab implementation
        return self._simplified_numbering(heavy_seq, light_seq)

    def _simplified_numbering(self,
                             heavy_seq: str,
                             light_seq: str) -> Tuple[np.ndarray, List[int]]:
        """Simplified numbering for Colab implementation."""
        # Initialize with gaps
        fixed_seq = ["-"] * (2 * self.num_positions_per_chain)

        # Fill in heavy chain (truncate or pad as needed)
        for i, aa in enumerate(heavy_seq[:self.num_positions_per_chain]):
            fixed_seq[i] = aa

        # Fill in light chain (truncate or pad as needed)
        for i, aa in enumerate(light_seq[:self.num_positions_per_chain]):
            fixed_seq[self.num_positions_per_chain + i] = aa

        # Convert to one-hot
        one_hot = self._to_one_hot(fixed_seq)

        # Get non-gap positions
        non_gap_positions = [i for i, aa in enumerate(fixed_seq) if aa != "-"]

        return one_hot, non_gap_positions

    def _to_one_hot(self, sequence: List[str]) -> np.ndarray:
        """Convert amino acid sequence to one-hot encoding."""
        # Amino acid vocabulary (20 amino acids + gap)
        aa_vocab = list("ACDEFGHIKLMNPQRSTVWY-")
        aa_to_idx = {aa: i for i, aa in enumerate(aa_vocab)}

        # Create one-hot encoding
        one_hot = np.zeros((len(sequence), len(aa_vocab)))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                one_hot[i, aa_to_idx[aa]] = 1
            else:
                # Use gap for unknown amino acids
                one_hot[i, aa_to_idx["-"]] = 1

        return one_hot
