"""
train.py
========
Training loop for the Transformer machine translation model.

Features:
  - Adam optimizer with warmup learning rate schedule (from the paper)
  - Label smoothing cross-entropy loss
  - Validation loss tracking + best model checkpointing
  - Experiment logging to JSON (for the required experiment log)
  - Early stopping

Usage:
    python train.py --config configs/base.yaml
    python train.py --d_model 256 --num_layers 3 --epochs 20   (small model for quick test)
"""

import os
import math
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

from models.transformer import Transformer
from data.dataset import get_dataloaders


# ─────────────────────────────────────────────
#  1. Label Smoothing Loss
# ─────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Instead of hard 0/1 targets, smoothing distributes a small probability
    (epsilon) to all other classes. This reduces overconfidence and improves
    generalization (Szegedy et al., 2016).

    Args:
        vocab_size:  number of output classes
        pad_idx:     index of <pad> — excluded from loss
        smoothing:   label smoothing factor (default 0.1 per the paper)
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        Args:
            logits: (batch * tgt_len, vocab_size) — raw model outputs
            target: (batch * tgt_len,) — ground truth token indices

        Returns:
            scalar loss (mean over non-pad tokens)
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_target[:, self.pad_idx] = 0  # zero out pad

        # Compute KL divergence loss
        loss = -(smooth_target * log_probs).sum(dim=-1)

        # Mask out padding positions
        non_pad = (target != self.pad_idx)
        loss = loss[non_pad].mean()
        return loss


# ─────────────────────────────────────────────
#  2. Warmup Learning Rate Scheduler
# ─────────────────────────────────────────────

class WarmupScheduler:
    """
    Learning rate schedule from "Attention Is All You Need":

        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Learning rate increases linearly for the first warmup_steps steps,
    then decays proportionally to the inverse square root of the step.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _get_lr(self):
        s = self.step_num
        return self.d_model ** (-0.5) * min(s ** (-0.5), s * self.warmup_steps ** (-1.5))


# ─────────────────────────────────────────────
#  3. One Epoch of Training
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    """
    Run one full pass over the training data.

    Teacher forcing: the model receives the ground-truth target tokens
    as input at each step (tgt[:, :-1]) and predicts the next token (tgt[:, 1:]).

    Args:
        grad_clip: gradient clipping threshold to prevent exploding gradients

    Returns:
        avg_loss: average loss per token
        avg_lr:   learning rate at the end of the epoch
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    current_lr = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: feed tgt[:-1] as input, predict tgt[1:] as output
        tgt_input  = tgt[:, :-1]   # (batch, tgt_len - 1)
        tgt_target = tgt[:, 1:]    # (batch, tgt_len - 1) — shifted by 1

        # Forward pass
        logits = model(src, tgt_input,
                       src_pad_idx=0,   # pad index from vocabulary (same for both)
                       tgt_pad_idx=0)

        # Reshape for loss: (batch * tgt_len, vocab_size) vs (batch * tgt_len,)
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # prevent exploding gradients
        current_lr = scheduler.step()
        optimizer.step()

        # Accumulate stats (count non-pad tokens for accurate avg)
        non_pad = (target_flat != 0).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens, current_lr


# ─────────────────────────────────────────────
#  4. Validation
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test set (no gradient updates).

    Returns:
        avg_loss: average loss per token
    """
    model.eval()
    total_loss   = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            logits = model(src, tgt_input, src_pad_idx=0, tgt_pad_idx=0)

            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_target.reshape(-1)

            loss = criterion(logits_flat, target_flat)

            non_pad = (target_flat != 0).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / total_tokens


# ─────────────────────────────────────────────
#  5. Main Training Script
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=args.batch_size,
        min_freq=args.min_freq
    )

    # ── Build model ──
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ── Loss, optimizer, scheduler ──
    criterion = LabelSmoothingLoss(len(tgt_vocab), pad_idx=tgt_vocab.pad_idx, smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)

    # ── Experiment log ──
    log = {
        "experiment_id": args.exp_id,
        "hyperparameters": vars(args),
        "model_params": num_params,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "epochs": []
    }

    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, current_lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        train_ppl = math.exp(min(train_loss, 20))  # perplexity = e^loss
        val_ppl   = math.exp(min(val_loss, 20))

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} (PPL {train_ppl:.1f}) | "
              f"Val Loss: {val_loss:.4f} (PPL {val_ppl:.1f}) | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Log this epoch
        log["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl":  round(train_ppl, 2),
            "val_loss":   round(val_loss, 4),
            "val_ppl":    round(val_ppl, 2),
            "lr":         round(current_lr, 8),
            "time_sec":   round(elapsed, 1)
        })

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"checkpoints/best_model_exp{args.exp_id}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args": vars(args)
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save experiment log
    log_path = f"logs/experiment_{args.exp_id}.json"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nExperiment log saved to: {log_path}")


# ─────────────────────────────────────────────
#  CLI Arguments
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer for EN-DE translation")

    # Experiment tracking
    parser.add_argument("--exp_id",       type=int,   default=1,    help="Experiment ID for logging")

    # Model hyperparameters
    parser.add_argument("--d_model",      type=int,   default=256,  help="Model dimension")
    parser.add_argument("--num_heads",    type=int,   default=8,    help="Number of attention heads")
    parser.add_argument("--num_layers",   type=int,   default=3,    help="Number of encoder/decoder layers")
    parser.add_argument("--d_ff",         type=int,   default=512,  help="Feed-forward inner dimension")
    parser.add_argument("--dropout",      type=float, default=0.1,  help="Dropout rate")

    # Training settings
    parser.add_argument("--epochs",       type=int,   default=20,   help="Number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=128,  help="Batch size")
    parser.add_argument("--warmup_steps", type=int,   default=4000, help="Warmup steps for LR schedule")
    parser.add_argument("--min_freq",     type=int,   default=2,    help="Min token frequency for vocab")

    args = parser.parse_args()
    main(args)
