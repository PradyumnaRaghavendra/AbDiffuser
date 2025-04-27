
### train_model.py

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

sys.path.insert(0, '/content')

from abdiffuser.models.apmixer import APMixer
from abdiffuser.models.diffusion import GaussianDiffusion, DiscreteDiffusion, AbDiffuser
from abdiffuser.models.projection import ResidueProjection
from abdiffuser.models.priors import ResidueFrequencyPrior
from abdiffuser.utils.data_loader import ProcessedOASDataset


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = total_pos_loss = total_aa_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        one_hot = batch['one_hot'].to(device)
        positions = batch['positions'].to(device)

        t = torch.randint(0, model.atom_diffusion.num_diffusion_steps, (positions.shape[0],), device=device)

        optimizer.zero_grad()

        try:
            loss_pos, loss_aa = model(positions, one_hot, t)
            loss = loss_pos + loss_aa

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pos_loss += loss_pos.item()
            total_aa_loss += loss_aa.item()

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_loss': f"{loss_pos.item():.4f}",
                'aa_loss': f"{loss_aa.item():.4f}"
            })

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                torch.cuda.empty_cache()
            else:
                raise e

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, total_pos_loss / max(1, len(dataloader)), total_aa_loss / max(1, len(dataloader))


def validate(model, dataloader, device):
    model.eval()
    total_loss = total_pos_loss = total_aa_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            one_hot = batch['one_hot'].to(device)
            positions = batch['positions'].to(device)

            t = torch.randint(0, model.atom_diffusion.num_diffusion_steps, (positions.shape[0],), device=device)

            loss_pos, loss_aa = model(positions, one_hot, t)
            loss = loss_pos + loss_aa

            total_loss += loss.item()
            total_pos_loss += loss_pos.item()
            total_aa_loss += loss_aa.item()

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, total_pos_loss / max(1, len(dataloader)), total_aa_loss / max(1, len(dataloader))


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(output_dir, 'latest.pt'))
    if epoch % 5 == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch}.pt'))
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best.pt'))
        print(f"Saved new best model with loss {loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train AbDiffuser model")
    parser.add_argument("--output_dir", type=str, default="abdiffuser/experiments/checkpoints")
    parser.add_argument("--oas_data", type=str, default="abdiffuser/data/processed/oas_aligned_synthetic.pkl")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim", type=int, default=320)
    parser.add_argument("--use_side_chains", action="store_true")
    parser.add_argument("--num_diffusion_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    dataset = ProcessedOASDataset(args.oas_data, args.max_examples)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ProcessedOASDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ProcessedOASDataset.collate_fn)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    apmixer = APMixer(num_layers=args.num_layers, dim=args.dim, use_side_chains=args.use_side_chains).to(device)
    atom_diffusion = GaussianDiffusion(noise_schedule='cosine', num_diffusion_steps=args.num_diffusion_steps)
    aa_diffusion = DiscreteDiffusion(num_classes=21, num_diffusion_steps=args.num_diffusion_steps)
    projection_layer = ResidueProjection(use_side_chains=args.use_side_chains)

    model = AbDiffuser(model=apmixer, atom_diffusion=atom_diffusion, aa_diffusion=aa_diffusion, projection_layer=projection_layer).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss, train_pos_loss, train_aa_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_pos_loss, val_aa_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, epoch, val_loss, args.output_dir, is_best)

    print("Training completed!")


if __name__ == "__main__":
    main()
