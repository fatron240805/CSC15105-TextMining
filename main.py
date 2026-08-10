import os
import torch

from src.preprocess import load_document, split_sentences
from src.embedding import load_embedding_model, encode_sentences
from src.similarity import cosine_similarity, get_top_k_matches
from src.alignment import align_matches, alignments_to_spans
from src.utils import save_predictions_text, save_predictions_xml

MODEL_NAME = "intfloat/e5-base-v2"

THRESHOLD = 0.86
TOP_K = 5

MAX_SUSPICIOUS_GAP = 2
MAX_SOURCE_GAP = 5
MIN_PATH_LENGTH = 2

SUSPICIOUS_PATH = (
    "dataset/validation-data/validation/"
    "susp/suspicious-document010237.txt"
)

SOURCE_PATH = (
    "dataset/validation-data/validation/"
    "src/source-document010237.txt"
)

TEXT_OUTPUT_PATH = "outputs/predictions.txt"
XML_OUTPUT_PATH = "outputs/predictions.xml"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Top-k: {TOP_K}")
    print(f"Max suspicious gap: {MAX_SUSPICIOUS_GAP}")
    print(f"Max source gap: {MAX_SOURCE_GAP}")
    print(f"Min path length: {MIN_PATH_LENGTH}")

    model = load_embedding_model(MODEL_NAME, device=device)

    suspicious_text = load_document(SUSPICIOUS_PATH)
    source_text = load_document(SOURCE_PATH)

    suspicious_sentences = split_sentences(suspicious_text)
    source_sentences = split_sentences(source_text)

    print(f"Suspicious sentences: {len(suspicious_sentences)}")
    print(f"Source sentences: {len(source_sentences)}")

    suspicious_embeddings = encode_sentences(
        suspicious_sentences,
        model,
        prefix="query:",
    )

    source_embeddings = encode_sentences(
        source_sentences,
        model,
        prefix="passage:",
    )

    print("Suspicious embedding:", suspicious_embeddings.shape)
    print("Source embedding:", source_embeddings.shape)

    similarity_matrix = cosine_similarity(
        suspicious_embeddings,
        source_embeddings,
    )

    print("Similarity matrix:", similarity_matrix.shape)

    matches = get_top_k_matches(
        similarity_matrix,
        threshold=THRESHOLD,
        top_k=TOP_K,
    )

    print(f"Candidate matches: {len(matches)}")

    alignments = align_matches(
        matches,
        max_suspicious_gap=MAX_SUSPICIOUS_GAP,
        max_source_gap=MAX_SOURCE_GAP,
        min_path_length=MIN_PATH_LENGTH,
    )

    print(f"Alignments: {len(alignments)}")

    predictions = alignments_to_spans(
        alignments,
        suspicious_sentences,
        source_sentences,
    )

    print(f"Predictions: {len(predictions)}")

    suspicious_filename = os.path.basename(SUSPICIOUS_PATH)
    source_filename = os.path.basename(SOURCE_PATH)

    save_predictions_text(
        predictions=predictions,
        suspicious_text=suspicious_text,
        source_text=source_text,
        output_path=TEXT_OUTPUT_PATH,
    )

    save_predictions_xml(
        predictions=predictions,
        output_path=XML_OUTPUT_PATH,
        suspicious_filename=suspicious_filename,
        source_filename=source_filename,
    )

    print(f"\nText result: {TEXT_OUTPUT_PATH}")
    print(f"XML result: {XML_OUTPUT_PATH}")


if __name__ == "__main__":
    main()