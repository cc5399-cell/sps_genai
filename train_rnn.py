"""Module 7 activity: train the LSTM text generator on The Count of Monte Cristo.

Run:  uv run python train_rnn.py [EPOCHS]
"""
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helper_lib.model import LSTMModel
from helper_lib.text_utils import TextDataset, build_vocab, encode, load_corpus, save_vocab, tokenize
from helper_lib.trainer import train_rnn_model
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

torch.manual_seed(42)
device = get_device()

tokens = tokenize(load_corpus())
vocab = build_vocab(tokens, vocab_size=10000)
encoded = encode(tokens, vocab)
print(f"Using device: {device} | tokens: {len(tokens):,} | vocabulary: {len(vocab):,} | epochs: {EPOCHS}")

train_loader = DataLoader(TextDataset(encoded), batch_size=64, shuffle=True)

model = LSTMModel(vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_rnn_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

model.to("cpu")
save_model(model, "models/lstm_text.pth")
save_vocab(vocab, "models/lstm_vocab.json")
