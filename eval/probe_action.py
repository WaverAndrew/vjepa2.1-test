"""
Phase 2: Train and evaluate an attentive probe for action anticipation.

Follows the V-JEPA 2.1 evaluation protocol (Appendix C.6):
  - Frozen encoder, only probe weights are trained
  - 32 frames @ 8fps, 384px, context ending 1s before action start
  - Focal loss (alpha=0.25, gamma=2.0)
  - Metric: mean-class recall@5 for verb, noun, action

Usage:
    # Train:
    python eval/probe_action.py \
        --mode train \
        --ego4d_root /data/ego4d \
        --model vit_giant \
        --output_dir outputs/probes/action_ant \
        --epochs 20 \
        --lr 1e-4

    # Evaluate:
    python eval/probe_action.py \
        --mode eval \
        --ego4d_root /data/ego4d \
        --probe_checkpoint outputs/probes/action_ant/best.pt \
        --model vit_giant
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vjepa21_lib.model.loader import load_encoder_from_hub
from vjepa21_lib.data.ego4d import Ego4DSTADataset
from vjepa21_lib.probing.attentive_probe import ActionAnticipationProbe, FocalLoss


# V-JEPA 2.1 encoder output dims
ENCODER_DIMS = {
    "vjepa2_1_vit_base_384":      768,
    "vjepa2_1_vit_large_384":    1024,
    "vjepa2_1_vit_giant_384":    1408,
    "vjepa2_1_vit_gigantic_384": 1664,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--ego4d_root", type=str, required=True)
    p.add_argument("--model", type=str, default="vjepa2_1_vit_giant_384",
                   choices=list(ENCODER_DIMS))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--probe_checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs/probes/action")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=384)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def extract_features_batch(encoder, frames: torch.Tensor, device: str) -> torch.Tensor:
    """Forward encoder on a batch of clips. frames: (B, T, C, H, W)"""
    B, T, C, H, W = frames.shape
    x = frames.permute(0, 2, 1, 3, 4).to(device)  # (B, C, T, H, W)
    feats = encoder(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    return feats  # (B, N, D)


def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Mean-class Recall@K following EK100 protocol."""
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(targets.unsqueeze(1).expand_as(topk))
    per_sample = correct.any(dim=1).float()

    # Mean-class: average recall per class (not per sample)
    num_classes = logits.shape[1]
    class_recall = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_recall.append(per_sample[mask].mean().item())
    return float(torch.tensor(class_recall).mean()) if class_recall else 0.0


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    enc_dim = ENCODER_DIMS[args.model]

    print(f"Loading encoder: {args.model}")
    encoder = load_encoder_from_hub(args.model).to(device)

    probe = ActionAnticipationProbe(encoder_dim=enc_dim, probe_dim=enc_dim).to(device)

    print("Loading datasets...")
    train_ds = Ego4DSTADataset(args.ego4d_root, split="train", num_frames=args.num_frames,
                                fps=args.fps, crop_size=args.crop_size)
    val_ds = Ego4DSTADataset(args.ego4d_root, split="val", num_frames=args.num_frames,
                              fps=args.fps, crop_size=args.crop_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_recall = 0.0

    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0.0

        for batch in train_loader:
            frames = batch["frames"]  # (B, T, C, H, W)
            verb_targets = batch["verb_class"].to(device)
            noun_targets = batch["noun_class"].to(device)

            feats = extract_features_batch(encoder, frames, device)  # (B, N, D)

            verb_logits, noun_logits = probe(feats)
            loss = criterion(verb_logits, verb_targets) + criterion(noun_logits, noun_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        probe.eval()
        all_verb_logits, all_noun_logits, all_verb_targets, all_noun_targets = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"]
                feats = extract_features_batch(encoder, frames, device)
                verb_logits, noun_logits = probe(feats)
                all_verb_logits.append(verb_logits.cpu())
                all_noun_logits.append(noun_logits.cpu())
                all_verb_targets.append(batch["verb_class"])
                all_noun_targets.append(batch["noun_class"])

        verb_rec = recall_at_k(torch.cat(all_verb_logits), torch.cat(all_verb_targets))
        noun_rec = recall_at_k(torch.cat(all_noun_logits), torch.cat(all_noun_targets))
        mean_rec = (verb_rec + noun_rec) / 2

        print(f"Epoch {epoch+1}/{args.epochs} | loss={total_loss/len(train_loader):.4f} "
              f"| verb_rec@5={verb_rec:.3f} | noun_rec@5={noun_rec:.3f} | mean={mean_rec:.3f}")

        if mean_rec > best_recall:
            best_recall = mean_rec
            torch.save(probe.state_dict(), output_dir / "best.pt")
            print(f"  -> New best saved (mean recall@5={best_recall:.3f})")

    print(f"\nTraining complete. Best mean recall@5: {best_recall:.3f}")


def evaluate(args):
    device = args.device
    enc_dim = ENCODER_DIMS[args.model]

    encoder = load_encoder_from_hub(args.model).to(device)
    probe = ActionAnticipationProbe(encoder_dim=enc_dim, probe_dim=enc_dim).to(device)

    assert args.probe_checkpoint, "Provide --probe_checkpoint for evaluation"
    probe.load_state_dict(torch.load(args.probe_checkpoint, map_location=device))
    probe.eval()

    val_ds = Ego4DSTADataset(args.ego4d_root, split="val", num_frames=args.num_frames,
                              fps=args.fps, crop_size=args.crop_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    all_verb_logits, all_noun_logits, all_verb_targets, all_noun_targets = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            frames = batch["frames"]
            feats = extract_features_batch(encoder, frames, device)
            verb_logits, noun_logits = probe(feats)
            all_verb_logits.append(verb_logits.cpu())
            all_noun_logits.append(noun_logits.cpu())
            all_verb_targets.append(batch["verb_class"])
            all_noun_targets.append(batch["noun_class"])

    verb_rec = recall_at_k(torch.cat(all_verb_logits), torch.cat(all_verb_targets))
    noun_rec = recall_at_k(torch.cat(all_noun_logits), torch.cat(all_noun_targets))

    print(f"Verb Recall@5:   {verb_rec:.4f}")
    print(f"Noun Recall@5:   {noun_rec:.4f}")
    print(f"Mean Recall@5:   {(verb_rec + noun_rec) / 2:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate(args)
