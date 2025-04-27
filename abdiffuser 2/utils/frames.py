"""
SE(3) frame utilities for achieving rotation and translation equivariance.
"""

import torch
import numpy as np

def compute_frames(x_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute SE(3) frames for protein positions using PCA.
    """
    batch_size, n_residues, atom_types, _ = x_pos.shape
    device = x_pos.device

    # Reshape to [batch, n_points, 3]
    positions = x_pos.reshape(batch_size, n_residues * atom_types, 3)

    # Compute centroids
    centroids = positions.mean(dim=1, keepdim=True)  # [batch, 1, 3]

    # Center positions
    centered = positions - centroids  # [batch, n_points, 3]

    # Compute covariance matrix
    cov = torch.bmm(centered.transpose(1, 2), centered)  # [batch, 3, 3]

    # Perform eigendecomposition
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    except:
        # Fallback for numerical stability
        cov = cov + 1e-4 * torch.eye(3, device=device).unsqueeze(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort eigenvectors by eigenvalues in descending order
    idx = torch.argsort(eigenvalues, dim=1, descending=True)
    eigenvectors = torch.gather(eigenvectors, 2,
                               idx.unsqueeze(1).expand(-1, 3, -1))

    # Create frames with different sign combinations
    frames = []
    for alpha in [-1, 1]:
        for beta in [-1, 1]:
            # First two axes with sign variations
            v1 = alpha * eigenvectors[:, :, 0]  # [batch, 3]
            v2 = beta * eigenvectors[:, :, 1]   # [batch, 3]

            # Third axis using cross product (right-hand rule)
            v3 = torch.cross(v1, v2, dim=1)     # [batch, 3]

            # Create rotation matrix
            R = torch.stack([v1, v2, v3], dim=2)  # [batch, 3, 3]

            # Create translation (centroid)
            t = centroids.squeeze(1)  # [batch, 3]

            # Combine into transformation matrix [R|t]
            transform = torch.cat([R, t.unsqueeze(2)], dim=2)  # [batch, 3, 4]
            frames.append(transform)

    # Stack all frames
    frames = torch.stack(frames, dim=1)  # [batch, 4, 3, 4]

    return frames
