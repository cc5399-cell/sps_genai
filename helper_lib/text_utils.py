"""Corpus, tokenizer and vocabulary helpers for the LSTM text generator (Module 6 RNN / Module 8 GPT notebooks)."""
import json
import os
import re
import urllib.request
from collections import Counter

import torch
from torch.utils.data import Dataset

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"  # The Count of Monte Cristo
SEQ_LEN = 30


def load_corpus(cache_path="data/monte_cristo.txt"):
    """Download the novel once, cache it, and return the text without the Gutenberg header/footer."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            raw_text = fh.read()
    else:
        with urllib.request.urlopen(GUTENBERG_URL, timeout=60) as resp:
            raw_text = resp.read().decode("utf-8")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(raw_text)
    start_idx = raw_text.find("Chapter 1")
    end_idx = raw_text.rfind("Chapter 5")
    return raw_text[start_idx:end_idx]


def tokenize(text):
    """Word-level tokens: drop punctuation, lowercase, split on whitespace."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).lower().split()


def build_vocab(tokens, vocab_size=10000):
    """Map the vocab_size - 2 most frequent words to ids 2.., plus <PAD> = 0 and <UNK> = 1."""
    counter = Counter(tokens)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode(tokens, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokens]


def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TextDataset(Dataset):
    """Item i is (tokens[i : i+SEQ_LEN], tokens[i+1 : i+SEQ_LEN+1]): predict every next word."""

    def __init__(self, data, seq_len=SEQ_LEN):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_len]),
                torch.tensor(self.data[idx + 1:idx + self.seq_len + 1]))
