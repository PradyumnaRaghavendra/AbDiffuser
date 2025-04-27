import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_schedule="cosine", num_diffusion_steps=200, predict_mode="x0", snr_weight=True, cov_matrix=None):
        super().__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.predict_mode = predict_mode
        self.snr_weight = snr_weight
        self.cov_matrix = cov_matrix

        if noise_schedule == "cosine":
            self.alphas, self.betas = self._cosine_schedule(num_diffusion_steps)
        else:
            self.alphas, self.betas = self._linear_schedule(num_diffusion_steps)

        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.snr = self.alpha_cumprod / (1 - self.alpha_cumprod)

    def _cosine_schedule(self, num_diffusion_steps, precision=1e-4):
        steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float32) / num_diffusion_steps
        alphas_cumprod = torch.cos((steps + precision) / (1 + precision) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1.0 - betas
        return alphas, betas

    def _linear_schedule(self, num_diffusion_steps):
        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        return alphas, betas

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_cumprod_t = self.alpha_cumprod.to(x_0.device)[t]
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        for _ in range(len(x_0.shape) - 1):
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def loss(self, model_fn, x_0, t=None, reduction="mean"):
        batch_size = x_0.shape[0]
        device = x_0.device
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        model_output = model_fn(x_t, t)
        target = x_0 if self.predict_mode == "x0" else noise
        loss = F.mse_loss(model_output, target, reduction="none")

        if self.snr_weight:
            snr_t = self.snr.to(device)[t]
            for _ in range(len(loss.shape) - 1):
                snr_t = snr_t.unsqueeze(-1)
            loss = loss * snr_t if self.predict_mode == "x0" else loss / snr_t

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss

class DiscreteDiffusion(nn.Module):
    def __init__(self, num_classes=21, num_diffusion_steps=200, position_specific_prior=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_diffusion_steps = num_diffusion_steps
        self.position_specific_prior = position_specific_prior
        self.target_distribution = torch.ones(num_classes) / num_classes
        alphas, betas = self._cosine_schedule(num_diffusion_steps)
        self.alphas = alphas
        self.betas = betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    def _cosine_schedule(self, num_diffusion_steps, precision=1e-4):
        steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float32) / num_diffusion_steps
        alphas_cumprod = torch.cos((steps + precision) / (1 + precision) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0, 0.999)
        alphas = 1.0 - betas
        return alphas, betas

    def q_sample(self, x_0, t, return_logits=False):
        batch_size, seq_len, _ = x_0.shape
        device = x_0.device
        alpha_t = self.alpha_cumprod.to(device)[t].view(-1, 1, 1).expand(-1, seq_len, -1)
        target_dist = self.target_distribution.to(device).view(1, 1, -1).expand(batch_size, seq_len, -1)
        identity = torch.eye(self.num_classes, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, self.num_classes, self.num_classes)
        q_t = alpha_t.unsqueeze(-1) * identity + (1 - alpha_t).unsqueeze(-1) * target_dist.unsqueeze(-2)
        if return_logits:
            flat_x_0 = x_0.reshape(-1, self.num_classes)
            flat_q_t = q_t.reshape(-1, self.num_classes, self.num_classes)
            flat_logits = torch.bmm(flat_x_0.unsqueeze(1), torch.log(flat_q_t + 1e-10)).squeeze(1)
            return flat_logits.reshape(batch_size, seq_len, self.num_classes), q_t
        else:
            flat_x_0 = x_0.reshape(-1, self.num_classes)
            flat_q_t = q_t.reshape(-1, self.num_classes, self.num_classes)
            flat_probs = torch.bmm(flat_x_0.unsqueeze(1), flat_q_t).squeeze(1)
            probs = flat_probs.reshape(batch_size, seq_len, self.num_classes)
            indices = torch.multinomial(probs.reshape(-1, self.num_classes), 1).reshape(batch_size, seq_len)
            x_t = F.one_hot(indices, self.num_classes).float()
            return x_t

    def loss(self, model_fn, x_0, t=None, reduction="mean"):
        batch_size = x_0.shape[0]
        device = x_0.device
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        x_t_logits, _ = self.q_sample(x_0, t, return_logits=True)
        x_t = F.gumbel_softmax(x_t_logits, tau=1.0, hard=True)
        pred_x0_logits = model_fn(x_t, t)
        loss = F.cross_entropy(pred_x0_logits.reshape(-1, self.num_classes), torch.argmax(x_0, dim=-1).reshape(-1), reduction="none")
        loss = loss.view(batch_size, -1)
        beta_t = 1 - self.alpha_cumprod.to(device)[t]
        beta_t = beta_t.view(-1, 1)
        loss = loss * beta_t
        return loss.mean() if reduction == "mean" else loss.sum()

class AbDiffuser(nn.Module):
    def __init__(self, model, atom_diffusion, aa_diffusion, projection_layer=None):
        super().__init__()
        self.model = model
        self.atom_diffusion = atom_diffusion
        self.aa_diffusion = aa_diffusion
        self.projection_layer = projection_layer

    def forward(self, x_pos, x_aa, t_pos=None, t_aa=None):
        batch_size = x_pos.shape[0]
        device = x_pos.device
        if t_pos is None:
            t_pos = torch.randint(0, self.atom_diffusion.num_diffusion_steps, (batch_size,), device=device)
        if t_aa is None:
            t_aa = t_pos
        if self.projection_layer is not None:
            x_pos = self.projection_layer(x_pos, x_aa)
        x_pos_t = self.atom_diffusion.q_sample(x_pos, t_pos)
        x_aa_t = self.aa_diffusion.q_sample(x_aa, t_aa)
        pred_pos, pred_aa = self.model(x_pos_t, x_aa_t, t_pos)
        loss_pos = self.atom_diffusion.loss(lambda x, t: pred_pos, x_pos, t_pos)
        loss_aa = self.aa_diffusion.loss(lambda x, t: pred_aa, x_aa, t_aa)
        return loss_pos, loss_aa
