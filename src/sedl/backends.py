"""Neural backends plus an explicitly labelled offline engineering demo."""

from collections import Counter
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    revision: str | None
    input_limit: int

    def count_tokens(self, text: str) -> int: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Summarizer(Protocol):
    name: str
    revision: str | None
    input_limit: int
    max_new_tokens: int

    def count_tokens(self, text: str) -> int: ...
    def summarize(self, text: str, *, max_new_tokens: int | None = None) -> str: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class HashEmbedder:
    """Lexical hashing for offline pipeline checks; NOT semantic embeddings."""

    name = "demo/lexical-hash-384"
    revision = None
    input_limit = 256

    def count_tokens(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))

    def encode(self, texts):
        vectors = np.zeros((len(texts), 384))
        for i, text in enumerate(texts):
            for word, frequency in Counter(re.findall(r"\w+", text.lower())).items():
                digest = hashlib.sha256(word.encode()).digest()
                vectors[i, int.from_bytes(digest[:4], "big") % 384] += frequency
        return normalize(vectors)


class ExtractiveDemo:
    """Echo selected evidence for a deterministic smoke run; NOT BART."""

    name = "demo/extractive-echo"
    revision = None
    input_limit = 1024
    max_new_tokens = 128

    def count_tokens(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))

    def summarize(self, text, *, max_new_tokens=None):
        if max_new_tokens is None:
            return text
        tokens = list(re.finditer(r"\w+|[^\w\s]", text))
        return text[:tokens[min(max_new_tokens, len(tokens)) - 1].end()] if tokens else ""


class SentenceTransformerEmbedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2", *,
                 device="cpu", revision=None):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device=device, revision=revision)
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", revision)
        self.tokenizer = self.model.tokenizer
        self.input_limit = self.model.max_seq_length - self.tokenizer.num_special_tokens_to_add()

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def encode(self, texts):
        if any(self.count_tokens(text) > self.input_limit for text in texts):
            raise ValueError("Embedding input exceeds model capacity; refusing silent truncation.")
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)


class BartSummarizer:
    def __init__(self, name="facebook/bart-large-cnn", *, device="cpu", revision=None,
                 max_new_tokens=128, num_beams=4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision)
        if self.model.config.model_type != "bart":
            raise ValueError("This research backend requires a BART checkpoint.")
        self.model.to(device).eval()
        self.device = device
        self.revision = getattr(self.model.config, "_commit_hash", revision)
        self.input_limit = (self.model.config.max_position_embeddings -
                            self.tokenizer.num_special_tokens_to_add())
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        if not 1 <= max_new_tokens < self.model.config.max_position_embeddings:
            raise ValueError("Invalid BART output token limit.")
        if num_beams < 1:
            raise ValueError("num_beams must be positive.")
        self.torch = torch

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def summarize(self, text, *, max_new_tokens=None):
        if self.count_tokens(text) > self.input_limit:
            raise ValueError("BART input exceeds model capacity; refusing silent truncation.")
        output_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= output_limit <= self.max_new_tokens:
            raise ValueError("Per-chunk output allowance must fit max_new_tokens")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=output_limit,
                                         min_length=0, num_beams=self.num_beams,
                                         do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False).strip()
