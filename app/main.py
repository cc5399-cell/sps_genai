from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embeddings import WordEmbedder

app = FastAPI(title="SPS GenAI API")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedder = WordEmbedder()


# ---------- Request / response models ----------

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["apple"])


class EmbeddingResponse(BaseModel):
    word: str
    dimension: int
    embedding: list[float]


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["apple"])
    word2: str = Field(..., min_length=1, examples=["car"])


class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float


# ---------- Endpoints ----------

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest):
    """Return the spaCy word embedding for the query word."""
    try:
        vector = embedder.embed(request.word)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err))
    return EmbeddingResponse(word=request.word, dimension=len(vector), embedding=vector)


@app.post("/similarity", response_model=SimilarityResponse)
def get_similarity(request: SimilarityRequest):
    """Return the cosine similarity between the embeddings of two words."""
    try:
        score = embedder.similarity(request.word1, request.word2)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err))
    return SimilarityResponse(word1=request.word1, word2=request.word2, similarity=score)
