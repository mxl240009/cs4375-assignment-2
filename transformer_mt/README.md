# Transformer for Machine Translation (EN → DE)

Machine Learning Course Project — Spring 2025  
**Topic:** Transformers for Machine Translation  
**Dataset:** Multi30k (English–German)

---

## Project Structure

```
transformer_mt/
├── models/
│   └── transformer.py      # Full Transformer: attention, encoder, decoder
├── data/
│   └── dataset.py          # Data loading, tokenization, vocabulary
├── train.py                # Training loop (warmup LR, label smoothing, logging)
├── evaluate.py             # BLEU score evaluation + greedy decoding
├── requirements.txt        # Python dependencies
└── logs/                   # Experiment JSON logs (auto-generated)
    └── experiment_1.json
    └── experiment_2.json
    └── ...
└── checkpoints/            # Saved model weights (auto-generated)
    └── best_model_exp1.pt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy language models

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## Running the Code

### Train (small model — fast, for testing)

```bash
python train.py \
  --exp_id 1 \
  --d_model 256 \
  --num_heads 8 \
  --num_layers 3 \
  --d_ff 512 \
  --dropout 0.1 \
  --epochs 20 \
  --batch_size 128 \
  --warmup_steps 4000
```

### Train (base model — full scale)

```bash
python train.py \
  --exp_id 2 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 6 \
  --d_ff 2048 \
  --dropout 0.1 \
  --epochs 30 \
  --batch_size 64 \
  --warmup_steps 4000
```

### Evaluate (compute BLEU score)

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model_exp1.pt \
  --exp_id 1
```

---

## Dataset

**Multi30k (EN–DE)**  
Hosted publicly via torchtext — no manual download needed.  
torchtext automatically fetches the dataset on first run.

- Training: ~29,000 sentence pairs  
- Validation: ~1,014 sentence pairs  
- Test: ~1,000 sentence pairs  

---

## Experiment Log Format

Each training run generates a JSON log at `logs/experiment_N.json`.  
After evaluation, BLEU scores are appended automatically.

Example:
```json
{
  "experiment_id": 1,
  "hyperparameters": {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 3,
    "d_ff": 512,
    "dropout": 0.1,
    "epochs": 20,
    "batch_size": 128,
    "warmup_steps": 4000
  },
  "model_params": 12500000,
  "epochs": [
    { "epoch": 1, "train_loss": 6.21, "train_ppl": 497.2, "val_loss": 5.98, "val_ppl": 395.1, "lr": 0.000012, "time_sec": 42.1 },
    { "epoch": 2, "train_loss": 5.44, "train_ppl": 230.4, "val_loss": 5.21, "val_ppl": 183.1, "lr": 0.000025, "time_sec": 41.8 }
  ],
  "val_bleu": 28.45,
  "test_bleu": 27.91
}
```

---

## Key Implementation Details

All core components are implemented from scratch in PyTorch (no `nn.Transformer`):

| Component | File | Description |
|---|---|---|
| Scaled Dot-Product Attention | `transformer.py` | Q·Kᵀ / √d_k → softmax → ·V |
| Multi-Head Attention | `transformer.py` | h parallel attention heads |
| Positional Encoding | `transformer.py` | Sinusoidal, fixed (not learned) |
| Encoder Layer | `transformer.py` | Self-Attention + FFN + LayerNorm |
| Decoder Layer | `transformer.py` | Masked Self-Attn + Cross-Attn + FFN |
| Label Smoothing Loss | `train.py` | ε=0.1, ignores padding |
| Warmup LR Schedule | `train.py` | d_model^-0.5 · min(step^-0.5, step·warmup^-1.5) |
| Greedy Decoding | `evaluate.py` | Argmax at each decoding step |
| BLEU Evaluation | `evaluate.py` | corpus_bleu via sacrebleu |

---

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).  
**TO DO: Attention is all we need.** Advances in Neural Information Processing Systems, 30.
