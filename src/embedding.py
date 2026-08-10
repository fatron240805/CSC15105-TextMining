from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "intfloat/e5-base-v2"

def encode_sentences(
    sentences: Sequence[dict],
    model: SentenceTransformer,
    prefix: str = "passage:",
) -> np.ndarray:
    '''
    Encode sentences into dense semantic embeddings.

    Args:
        sentences: Output from preprocess.split_sentences().
        model: SentenceTransformer model.
        prefix: Use "query" for suspisous and "passage" for source
    '''
    texts = [
        f"{prefix} {sentence['text']}"
        for sentence in sentences
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)

def load_embedding_model(model_name: str = MODEL_NAME, device: str | None = None) -> SentenceTransformer:
    '''
    Load the sentence embedding model.
    '''
    return SentenceTransformer(
        model_name,
        device=device,
    )