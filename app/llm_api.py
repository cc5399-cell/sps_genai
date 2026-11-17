"""Module 9: question answering with GPT-2 fine-tuned on the Nectar dataset."""
from pathlib import Path

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

FINETUNED_DIR = Path(__file__).resolve().parents[1] / "models" / "gpt2-qa"
BASE_MODEL = "openai-community/gpt2"

router = APIRouter(tags=["LLM"])


class LLMGenerationRequest(BaseModel):
    start_word: str = Field(..., min_length=1, examples=["What is machine learning?"],
                            description="The question to answer")
    length: int = Field(60, ge=1, le=300, description="Maximum number of new tokens")


class LLMTextGenerator:
    def __init__(self):
        # Use the fine-tuned model when it exists; otherwise fall back to the base GPT-2
        self.source = "fine-tuned" if FINETUNED_DIR.exists() else "base"
        path = str(FINETUNED_DIR) if FINETUNED_DIR.exists() else BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.eval()

    def generate(self, question, max_new_tokens=60):
        """Wrap the question in the Nectar prompt format and sample an answer."""
        prompt = f"Human: {question.strip()}\n\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.split("Human:")[0].strip()   # stop if the model starts a new turn


_generator = None


def get_generator():
    """Load the model on the first request rather than at startup (it is large)."""
    global _generator
    if _generator is None:
        _generator = LLMTextGenerator()
    return _generator


@router.post("/generate_with_llm")
def generate_with_llm(request: LLMGenerationRequest):
    """Answer a question with the fine-tuned GPT-2."""
    generator = get_generator()
    generated_text = generator.generate(request.start_word, request.length)
    return {"generated_text": generated_text, "model": generator.source}
