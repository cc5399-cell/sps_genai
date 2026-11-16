"""Module 7: text generation endpoint using the LSTM trained on The Count of Monte Cristo."""
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from helper_lib.model import LSTMModel
from helper_lib.text_utils import load_vocab, tokenize

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

router = APIRouter(tags=["RNN"])


class RNNGenerationRequest(BaseModel):
    start_word: str = Field(..., min_length=1, examples=["the count"])
    length: int = Field(20, ge=1, le=200, description="Number of words to generate")
    temperature: float = Field(1.0, gt=0, le=2.0, description="Lower = safer, higher = more random")


class RNNTextGenerator:
    def __init__(self, models_dir=MODELS_DIR):
        self.vocab = load_vocab(models_dir / "lstm_vocab.json")
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.model = LSTMModel(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(models_dir / "lstm_text.pth", map_location="cpu"))
        self.model.eval()

    def generate(self, seed_text, length=20, temperature=1.0):
        """Feed the seed words, then sample one next word at a time from the softmax."""
        words = tokenize(seed_text)
        if not words:
            raise ValueError("start_word must contain at least one letter or digit.")
        input_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        x = torch.tensor(input_ids).unsqueeze(0)
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                output, hidden = self.model(x, hidden)
                logits = output[0, -1] / temperature
                logits[[self.vocab["<PAD>"], self.vocab["<UNK>"]]] = float("-inf")  # never emit these
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()
                words.append(self.inv_vocab[next_id])
                x = torch.tensor([[next_id]])  # the hidden state already carries the earlier words

        return " ".join(words)


try:
    rnn_generator = RNNTextGenerator()
except FileNotFoundError:
    rnn_generator = None  # weights missing: run `uv run python train_rnn.py`


@router.post("/generate_with_rnn")
def generate_with_rnn(request: RNNGenerationRequest):
    """Generate text word by word with the LSTM language model."""
    if rnn_generator is None:
        raise HTTPException(status_code=503, detail="LSTM weights not found. Run: uv run python train_rnn.py")
    try:
        generated_text = rnn_generator.generate(request.start_word, request.length, request.temperature)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return {"generated_text": generated_text}
