import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
import spacy


# ─────────────────────────────────────────────
#  Special tokens (shared convention)
# ─────────────────────────────────────────────

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"  # start of sequence
EOS_TOKEN = "<eos>"  # end of sequence

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ─────────────────────────────────────────────
#  Vocabulary
# ─────────────────────────────────────────────

class Vocabulary:
    """
    Maps tokens ↔ integer indices.

    Args:
        min_freq: tokens appearing fewer times than this are mapped to <unk>
    """

    def __init__(self, min_freq: int = 2):
        self.min_freq  = min_freq
        self.token2idx = {}
        self.idx2token = {}

        # Reserve indices 0–3 for special tokens
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2idx[tok] = i
            self.idx2token[i]   = tok

    @property
    def pad_idx(self): return self.token2idx[PAD_TOKEN]
    @property
    def unk_idx(self): return self.token2idx[UNK_TOKEN]
    @property
    def sos_idx(self): return self.token2idx[SOS_TOKEN]
    @property
    def eos_idx(self): return self.token2idx[EOS_TOKEN]

    def build(self, token_lists):
        """
        Build vocabulary from a list of token lists.

        Args:
            token_lists: list of lists of strings  e.g. [["a", "cat"], ["the", "cat"]]
        """
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

    def encode(self, tokens):
        """Convert list of tokens to list of integer indices."""
        return [self.token2idx.get(tok, self.unk_idx) for tok in tokens]

    def decode(self, indices):
        """Convert list of indices back to tokens (skips special tokens)."""
        return [self.idx2token.get(i, UNK_TOKEN) for i in indices
                if i not in (self.pad_idx, self.sos_idx, self.eos_idx)]

    def __len__(self):
        return len(self.token2idx)


# ─────────────────────────────────────────────
#  Translation Dataset
# ─────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Wraps raw sentence pairs into a PyTorch Dataset.
    Encodes tokens to indices and wraps each sequence with <sos> and <eos>.

    Args:
        pairs:    list of (src_tokens, tgt_tokens) tuples
        src_vocab: Vocabulary for source language
        tgt_vocab: Vocabulary for target language
    """

    def __init__(self, pairs, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.pairs     = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]

        # Encode: add <sos> at start and <eos> at end
        src_ids = [self.src_vocab.sos_idx] + self.src_vocab.encode(src_tokens) + [self.src_vocab.eos_idx]
        tgt_ids = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(tgt_tokens) + [self.tgt_vocab.eos_idx]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# ─────────────────────────────────────────────
#  Collate Function (Padding)
# ─────────────────────────────────────────────

def make_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    """
    Returns a collate_fn that pads sequences in a batch to the same length.
    Shorter sequences are padded with <pad> on the right.
    """
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
        return src_padded, tgt_padded
    return collate_fn


# ─────────────────────────────────────────────
#  Main Data Loading Function
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128, min_freq: int = 2, max_len: int = 100):
    """
    Load Multi30k EN-DE dataset, build vocabularies, and return DataLoaders.

    Args:
        batch_size: number of sentence pairs per batch
        min_freq:   minimum token frequency to include in vocabulary
        max_len:    discard sentence pairs longer than this (reduces outlier noise)

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """

    # Load spacy tokenizers (must have these models installed)
    # Run: python -m spacy download en_core_web_sm
    # Run: python -m spacy download de_core_news_sm
    print("Loading spacy tokenizers...")
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")

    def tokenize_en(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def tokenize_de(text):
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

    # ── Load raw data from torchtext ──
    print("Loading Multi30k dataset...")
    train_data = list(Multi30k(split='train', language_pair=('en', 'de')))
    val_data   = list(Multi30k(split='valid', language_pair=('en', 'de')))
    test_data  = list(Multi30k(split='test',  language_pair=('en', 'de')))

    # ── Tokenize all splits ──
    def tokenize_pairs(data, max_len):
        pairs = []
        for src_text, tgt_text in data:
            src_toks = tokenize_en(src_text)
            tgt_toks = tokenize_de(tgt_text)
            # Filter out very long sentences
            if len(src_toks) <= max_len and len(tgt_toks) <= max_len:
                pairs.append((src_toks, tgt_toks))
        return pairs

    print("Tokenizing...")
    train_pairs = tokenize_pairs(train_data, max_len)
    val_pairs   = tokenize_pairs(val_data,   max_len)
    test_pairs  = tokenize_pairs(test_data,  max_len)

    # ── Build vocabularies from training data only ──
    print("Building vocabularies...")
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)

    src_vocab.build([src for src, _ in train_pairs])
    tgt_vocab.build([tgt for _, tgt in train_pairs])

    print(f"  Source vocab size: {len(src_vocab)}")
    print(f"  Target vocab size: {len(tgt_vocab)}")
    print(f"  Train pairs: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # ── Create Dataset and DataLoader objects ──
    collate_fn = make_collate_fn(src_vocab.pad_idx, tgt_vocab.pad_idx)

    train_loader = DataLoader(
        TranslationDataset(train_pairs, src_vocab, tgt_vocab),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        TranslationDataset(val_pairs, src_vocab, tgt_vocab),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        TranslationDataset(test_pairs, src_vocab, tgt_vocab),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
