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
