"""Module 9 activity: fine-tune GPT-2 (openai-community/gpt2) for question answering on Nectar.

Run:  uv run python finetune_gpt2.py [EPOCHS]
"""
import math
import sys

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper_lib.utils import get_device

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "models/gpt2-qa"
N_TRAIN, N_TEST = 2000, 200   # small subset of Nectar's 180k prompts, so training takes minutes
MAX_LENGTH = 256              # tokens kept per example
BATCH_SIZE = 8
LEARNING_RATE = 5e-5


def to_training_text(example, eos_token):
    """Nectar prompts look like 'Human: ...\\n\\nAssistant:'. Append the top-ranked answer."""
    best = min(example["answers"], key=lambda a: a["rank"])
    return example["prompt"].strip() + " " + best["answer"].strip() + eos_token


def make_collate_fn(tokenizer):
    """Tokenize a batch, pad it to its longest example, and mask the padding out of the loss."""
    def collate(texts):
        batch = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100   # -100 = ignored by the loss
        batch["labels"] = labels
        return batch
    return collate


def evaluate(model, data_loader, device):
    """Average next-token loss on the test set."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
    return total / len(data_loader)


def train(model, train_loader, test_loader, optimizer, device, epochs):
    """Training loop from the Module 9 GPT notebook, applied to the pre-trained model."""
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss   # passing labels makes the model compute the loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        test_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: test loss = {test_loss:.4f} (perplexity {math.exp(test_loss):.1f})")
    return model


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device} | model: {MODEL_NAME} | epochs: {epochs}")

    # 1. Prepare the data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no padding token
    stream = load_dataset("berkeley-nest/Nectar", split="train", streaming=True)  # no full download
    texts = [to_training_text(ex, tokenizer.eos_token) for ex, _ in zip(stream, range(N_TRAIN + N_TEST))]
    collate = make_collate_fn(tokenizer)
    train_loader = DataLoader(texts[:N_TRAIN], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(texts[N_TRAIN:], batch_size=BATCH_SIZE, collate_fn=collate)

    # 2. Get the base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    baseline = evaluate(model, test_loader, device)
    print(f"Before fine-tuning: test loss = {baseline:.4f} (perplexity {math.exp(baseline):.1f})")

    # 3. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train(model, train_loader, test_loader, optimizer, device, epochs)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
