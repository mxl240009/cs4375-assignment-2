import json
import torch
import argparse
import sacrebleu

from models.transformer import Transformer
from data.dataset import get_dataloaders


# ─────────────────────────────────────────────
#  Greedy Decoding
# ─────────────────────────────────────────────

def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=50):
    """
    Generate a translation using greedy search:
      - At each step, pick the token with the highest log-probability.
      - Stop when <eos> is generated or max_len is reached.

    This is simpler than beam search but fast enough for evaluation.

    Args:
        src: (1, src_len) — single source sentence as token indices

    Returns:
        List of predicted token strings (without special tokens)
    """
    model.eval()
    src = src.to(device)

    with torch.no_grad():
        # Encode the source sentence once
        src_mask   = model.make_src_mask(src, src_vocab.pad_idx)
        enc_output = model.encoder(src, src_mask)

        # Initialize decoder input with <sos>
        tgt_ids = [tgt_vocab.sos_idx]

        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            tgt_mask   = model.make_tgt_mask(tgt_tensor, tgt_vocab.pad_idx)

            logits = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)

            # Pick the most likely next token (greedy)
            next_token = logits[0, -1, :].argmax().item()
            tgt_ids.append(next_token)

            # Stop at <eos>
            if next_token == tgt_vocab.eos_idx:
                break

    # Decode indices to tokens (decode() already strips special tokens)
    return tgt_vocab.decode(tgt_ids)


# ─────────────────────────────────────────────
#  BLEU Score Evaluation
# ─────────────────────────────────────────────

def compute_bleu(model, loader, src_vocab, tgt_vocab, device, num_samples=None):
    """
    Compute corpus-level BLEU score on the given DataLoader.

    sacrebleu computes BLEU against detokenized reference strings,
    which is the standard approach for reporting machine translation results.

    Args:
        num_samples: if set, evaluate on only the first N batches (for speed)

    Returns:
        bleu_score (float), hypotheses (list[str]), references (list[str])
    """
    model.eval()
    hypotheses = []  # model predictions
    references = []  # ground-truth translations

    for i, (src_batch, tgt_batch) in enumerate(loader):
        if num_samples and i >= num_samples:
            break

        for j in range(src_batch.size(0)):
            # Decode single sentence
            src = src_batch[j].unsqueeze(0)  # (1, src_len)
            tgt = tgt_batch[j]               # (tgt_len,)

            # Generate prediction
            pred_tokens = greedy_decode(model, src, src_vocab, tgt_vocab, device)
            ref_tokens  = tgt_vocab.decode(tgt.tolist())

            hypotheses.append(" ".join(pred_tokens))
            references.append(" ".join(ref_tokens))

    # sacrebleu: references must be list of lists
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score, hypotheses, references


# ─────────────────────────────────────────────
#  Main Evaluation Script
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    model_args = ckpt["args"]

    # ── Rebuild model with saved hyperparameters ──
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_args["d_model"],
        num_heads=model_args["num_heads"],
        num_layers=model_args["num_layers"],
        d_ff=model_args["d_ff"],
        dropout=0.0  # disable dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ── Load data ──
    _, val_loader, test_loader, _, _ = get_dataloaders(
        batch_size=model_args["batch_size"],
        min_freq=model_args["min_freq"]
    )

    # ── Evaluate ──
    print("\nComputing BLEU on validation set...")
    val_bleu, val_hyps, val_refs = compute_bleu(
        model, val_loader, src_vocab, tgt_vocab, device
    )
    print(f"  Val BLEU:  {val_bleu:.2f}")

    print("\nComputing BLEU on test set...")
    test_bleu, test_hyps, test_refs = compute_bleu(
        model, test_loader, src_vocab, tgt_vocab, device
    )
    print(f"  Test BLEU: {test_bleu:.2f}")

    # ── Print sample translations ──
    print("\n── Sample Translations (first 5) ──")
    for i in range(min(5, len(test_hyps))):
        print(f"  Reference : {test_refs[i]}")
        print(f"  Hypothesis: {test_hyps[i]}")
        print()

    # ── Append BLEU to experiment log ──
    log_path = f"logs/experiment_{args.exp_id}.json"
    try:
        with open(log_path, "r") as f:
            log = json.load(f)
    except FileNotFoundError:
        log = {}

    log["val_bleu"]  = round(val_bleu, 2)
    log["test_bleu"] = round(test_bleu, 2)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"BLEU scores saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Transformer translation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--exp_id",     type=int, default=1,     help="Experiment ID for logging")
    args = parser.parse_args()
    main(args)
