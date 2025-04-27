"""
Physics-informed residue projection for AbDiffuser.
"""

import torch
import torch.nn as nn
import numpy as np

class ResidueProjection(nn.Module):
    """Physics-informed residue projection layer."""

    def __init__(self, use_side_chains=False):
        """Initialize the residue projection layer."""
        super().__init__()

        # Backbone bond lengths
        self.backbone_bond_lengths = {
            "N-CA": 1.458,   # N-CA bond
            "CA-C": 1.523,   # CA-C bond
            "CA-CB": 1.521,  # CA-CB bond
        }

        self.use_side_chains = use_side_chains

        # Create backbone template
        backbone_template = self._create_backbone_template()
        self.register_buffer("backbone_template", backbone_template)

    def _create_backbone_template(self):
        """Create an idealized backbone template with atoms N, CA, C, CB."""
        # Create an idealized backbone in a standard orientation
        template = torch.zeros((4, 3))

        # Nitrogen (N)
        template[0] = torch.tensor([0.0, 0.0, 0.0])

        # Alpha Carbon (CA)
        template[1] = torch.tensor([self.backbone_bond_lengths["N-CA"], 0.0, 0.0])

        # Carbon (C)
        ca_c_angle = np.radians(110)  # CA-C bond angle
        template[2] = torch.tensor([
            template[1, 0] - self.backbone_bond_lengths["CA-C"] * np.cos(ca_c_angle),
            self.backbone_bond_lengths["CA-C"] * np.sin(ca_c_angle),
            0.0
        ])

        # Beta Carbon (CB) - placed to form tetrahedral angle with N, CA, C
        cb_angle = np.radians(109.5)  # Tetrahedral angle
        template[3] = torch.tensor([
            template[1, 0] - self.backbone_bond_lengths["CA-CB"] * np.cos(cb_angle),
            template[1, 1] - self.backbone_bond_lengths["CA-CB"] * np.sin(cb_angle) * np.cos(np.pi/3),
            self.backbone_bond_lengths["CA-CB"] * np.sin(cb_angle) * np.sin(np.pi/3)
        ])

        return template

    def _kabsch_algorithm(self, P, Q):
        """Kabsch algorithm for optimal rotation between two sets of points."""
        # Center the points
        p_mean = torch.mean(P, dim=0)
        q_mean = torch.mean(Q, dim=0)

        P_centered = P - p_mean
        Q_centered = Q - q_mean

        # Compute covariance matrix
        H = P_centered.t() @ Q_centered

        # Singular value decomposition
        try:
            U, S, V = torch.linalg.svd(H)

            # Ensure proper rotation (with determinant 1)
            det = torch.det(V @ U.t())
            correction = torch.eye(3, device=P.device)
            correction[-1, -1] = det

            # Calculate rotation matrix
            R = V @ correction @ U.t()

            # Calculate translation
            t = q_mean - R @ p_mean

            return R, t
        except:
            # Fallback for numerical stability
            return torch.eye(3, device=P.device), torch.zeros(3, device=P.device)

    def forward(self, positions, aa_types):
        """Project atom positions to respect physical constraints."""
        batch_size, n_residues, n_atoms, _ = positions.shape
        device = positions.device

        # Get template
        template = self.backbone_template.to(device)

        # Reshape batch and residue dimensions for parallel processing
        positions_flat = positions.reshape(-1, n_atoms, 3)

        # Initialize output tensor
        projected_flat = torch.zeros_like(positions_flat)

        # Process each residue
        for i in range(batch_size * n_residues):
            # Get atoms for the current residue
            curr_pos = positions_flat[i]

            # Only process if positions are not all zeros
            if not torch.all(curr_pos == 0):
                try:
                    # Find optimal rotation and translation using Kabsch algorithm
                    R, t = self._kabsch_algorithm(curr_pos, template)

                    # Apply transformation to template
                    projected_flat[i] = (R @ template.t()).t() + t
                except:
                    # Fallback in case of numerical issues
                    projected_flat[i] = curr_pos
            else:
                # Keep zeros for missing residues
                projected_flat[i] = curr_pos

        # Reshape back to original dimensions
        projected = projected_flat.reshape(batch_size, n_residues, n_atoms, 3)

        return projected
