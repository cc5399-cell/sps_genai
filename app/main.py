import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.classifier import ImageClassifier
from app.embeddings import WordEmbedder
from app.gan_generator import DigitGenerator

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

try:
    classifier = ImageClassifier()
except FileNotFoundError:
    classifier = None  # weights missing: run `uv run python train_classifier.py`

try:
    digit_generator = DigitGenerator()
except FileNotFoundError:
    digit_generator = None  # weights missing: run `uv run python train_mnist_gan.py`


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


class ClassPrediction(BaseModel):
    label: str
    probability: float


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_predictions: list[ClassPrediction]


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


@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(..., description="An image file (PNG or JPEG)")):
    """Classify an uploaded image into one of the 10 CIFAR-10 classes."""
    if classifier is None:
        raise HTTPException(status_code=503,
                            detail="Classifier weights not found. Run: uv run python train_classifier.py")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return classifier.predict(image)


@app.get("/classify/model")
def classifier_info():
    """Describe the CNN behind /classify (architecture, parameter count, classes)."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier weights not found.")
    return classifier.describe()


@app.get("/gan/digits", response_class=Response,
         responses={200: {"content": {"image/png": {}}, "description": "A grid of generated digits"}})
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate (1-64)"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    """Generate new hand-written digits with the GAN trained on MNIST and return them as a PNG grid."""
    if digit_generator is None:
        raise HTTPException(status_code=503,
                            detail="GAN weights not found. Run: uv run python train_mnist_gan.py")
    return Response(content=digit_generator.generate_png(num_images, seed), media_type="image/png")


@app.get("/gan/model")
def gan_info():
    """Describe the GAN behind /gan/digits (generator and discriminator architecture)."""
    if digit_generator is None:
        raise HTTPException(status_code=503, detail="GAN weights not found.")
    return digit_generator.describe()
