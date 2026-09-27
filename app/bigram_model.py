"""Bigram text generator, adapted from Module 2 Practical 2 (Word Sampling)."""
import random
import re
from collections import Counter, defaultdict


def simple_tokenizer(text, frequency_threshold=None):
    """Lowercase the text and split it into words.

    If frequency_threshold is set, drop words that appear fewer times than that.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    return [token for token in tokens if word_counts[token] >= frequency_threshold]


class BigramModel:
    def __init__(self, corpus, frequency_threshold=None):
        """corpus is a list of strings, like the one in main.py."""
        text = " ".join(corpus)
        self.vocab, self.bigram_probs = self.analyze_bigrams(text, frequency_threshold)

    @staticmethod
    def analyze_bigrams(text, frequency_threshold=None):
        """Compute p(w2 | w1) = C(w1, w2) / C(w1)."""
        words = simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, length=20):
        """Generate text by repeatedly sampling the next word from bigram probabilities."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # no known continuation, stop early
                break
            current_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(current_word)

        return " ".join(generated_words)
