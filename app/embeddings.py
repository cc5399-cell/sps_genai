"""Word embeddings with spaCy, adapted from Module 2 Practical 3 (Word Embeddings)."""
import spacy


class WordEmbedder:
    def __init__(self, model_name: str = "en_core_web_lg"):
        # Load once at startup; the large model ships with 300-dimensional word vectors.
        self.nlp = spacy.load(model_name)

    def _doc(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")
        doc = self.nlp(text)
        if all(token.is_oov for token in doc):
            raise ValueError(f"No embedding available for '{text}'.")
        return doc

    def embed(self, text: str) -> list[float]:
        """Return the embedding for a word.

        For multi-word input, spaCy averages the word vectors (a sentence embedding).
        """
        return self._doc(text).vector.tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between the embeddings of two words or phrases."""
        return float(self._doc(text1).similarity(self._doc(text2)))
