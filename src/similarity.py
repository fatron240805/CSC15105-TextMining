import numpy as np

def cosine_similarity(
    suspicious_embeddings: np.ndarray,
    source_embeddings: np.ndarray,
) -> np.ndarray:
    '''
    Compute cosine similarity between all suspicious and source sentences.
    Embeddings are expected to be normalized.
    '''
    return suspicious_embeddings @ source_embeddings.T

def get_top_k_matches(
    similarity_matrix: np.ndarray,
    threshold: float,
    top_k: int = 5,
) -> list[dict]:
    '''
    Get top-k source sentence matches for each suspicious sentence with given threshold
    '''
    matches = []

    num_suspicious = similarity_matrix.shape[0]

    for susp_idx in range(num_suspicious):
        scores = similarity_matrix[susp_idx]

        top_indices = np.argsort(scores)[-top_k:][::-1]

        for source_idx in top_indices:
            score = float(scores[source_idx])

            if score > threshold:
                matches.append({
                    "suspicious_idx": susp_idx,
                    "source_idx": int(source_idx),
                    "score": score,
                })

    return matches