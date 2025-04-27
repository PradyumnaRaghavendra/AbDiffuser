import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Add project directory to path
sys.path.append('/content/abdiffuser')

from utils.frames import compute_frames

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim_out
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim // 2, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        return self.linear2(x)

class APMixerBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, equivariant_dim=None):
        super().__init__()
        self.dim = dim
        self.equivariant_dim = equivariant_dim or dim // 2
        self.col_norm = LayerNorm(dim)
        self.col_mlp = MLP(dim, dim, hidden_dim)
        self.row_norm = LayerNorm(dim)
        self.row_mlp = MLP(dim, dim, hidden_dim)
        self.merge = nn.Linear(dim, dim)

    def forward(self, x, frames=None):
        batch_size, n_residues, dim = x.shape
        x_eq = x[..., :self.equivariant_dim]
        x_inv = x[..., self.equivariant_dim:]

        x = torch.cat([x_eq, x_inv], dim=-1)
        x = self.merge(x)

        residual = x
        x = self.col_norm(x)
        x = self.col_mlp(x)
        x = residual + x

        residual = x
        x = self.row_norm(x)
        x = self.row_mlp(x)  # ❗ No transpose
        x = residual + x

        return x, frames

class APMixer(nn.Module):
    def __init__(
        self,
        num_layers=3,
        dim=320,
        hidden_dim=None,
        num_residues=149,
        atom_types=4,
        aa_types=21,
        use_side_chains=False,
        equivariant_dim=None,
        timestep_embedding_dim=32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.hidden_dim = hidden_dim or 2 * dim
        self.num_residues = num_residues
        self.atom_types = atom_types
        self.aa_types = aa_types
        self.use_side_chains = use_side_chains
        self.equivariant_dim = equivariant_dim or dim // 2
        self.timestep_embedding_dim = timestep_embedding_dim

        input_dim = aa_types + atom_types * 3 + timestep_embedding_dim

        self.embedding = MLP(input_dim, dim)

        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(timestep_embedding_dim, timestep_embedding_dim),
        )

        self.blocks = nn.ModuleList([
            APMixerBlock(dim, hidden_dim, equivariant_dim)
            for _ in range(num_layers)
        ])

        self.atom_pos_head = MLP(dim, atom_types * 3)
        self.aa_type_head = MLP(dim, aa_types)

    def forward(self, x_pos, x_aa, t):
        batch_size, n_residues, atom_types, _ = x_pos.shape
        device = x_pos.device

        frames = compute_frames(x_pos)
        x_pos_flat = x_pos.reshape(batch_size, n_residues, -1)

        t_emb = self.timestep_embedding(t.unsqueeze(1).float())
        t_emb = t_emb.unsqueeze(1).expand(-1, n_residues, -1)

        x = torch.cat([x_pos_flat, x_aa, t_emb], dim=-1)

        x = self.embedding(x)

        for block in self.blocks:
            x, frames = block(x, frames)

        pred_pos = self.atom_pos_head(x).reshape(batch_size, n_residues, atom_types, 3)
        pred_aa = self.aa_type_head(x)

        return pred_pos, pred_aa
