# SPS GenAI API

FastAPI service with a bigram text generator (Module 3) and spaCy word embeddings (Assignment 1).

## Run with Docker

```
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Interactive docs: http://127.0.0.1:8000/docs

## Run locally with uv

```
uv sync
uv run fastapi dev app/main.py
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | – | health check |
| POST | `/generate` | `{"start_word": "bigram", "length": 5}` | bigram-generated text |
| POST | `/embedding` | `{"word": "apple"}` | 300-dim spaCy (`en_core_web_lg`) embedding |
| POST | `/similarity` | `{"word1": "apple", "word2": "car"}` | cosine similarity of the two embeddings |

Example:

```
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

Unknown words return HTTP 404.

## Assignment 2: CNN image classifier

`helper_lib/model.py` → `AssignmentCNN` implements the specified architecture
(64x64x3 input → Conv 16 → ReLU → MaxPool → Conv 32 → ReLU → MaxPool → Flatten → FC 100 → ReLU → FC 10).
It is trained on CIFAR-10 (resized to 64x64) by `train_classifier.py`; the trained weights are
committed in `models/cnn_classifier.pth`, so the API works without retraining.

### Run

```
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/classify` | multipart form, field `file` = image (PNG/JPEG) | predicted CIFAR-10 class, confidence, top-3 |
| GET | `/classify/model` | – | architecture, parameter count, classes |

Example (sample CIFAR-10 test images are in `samples/`):

```
curl -X POST http://127.0.0.1:8000/classify -F "file=@samples/1_ship.png"
```

Retrain the model: `uv run python train_classifier.py 10`

## Assignment 3: GAN for hand-written digits

`helper_lib/model.py` → `MNISTGenerator`, `MNISTDiscriminator` (wrapped in `MNISTGAN`) implement the specified
architecture. `train_mnist_gan.py` trains it on MNIST with `train_standard_gan` (`helper_lib/trainer.py`);
the trained weights are committed in `models/mnist_gan.pth`, so the API works without retraining.

| Method | Path | Parameters | Returns |
|---|---|---|---|
| GET | `/gan/digits` | `num_images` (1-64, default 16), `seed` (optional) | PNG grid of generated digits |
| GET | `/gan/model` | – | generator and discriminator architecture |

Example:

```
curl "http://127.0.0.1:8000/gan/digits?num_images=16&seed=42" --output digits.png
```

Retrain the model: `uv run python train_mnist_gan.py 15`
