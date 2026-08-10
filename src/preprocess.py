from pathlib import Path
import re
import spacy

# Load English sentence segmentation model
nlp = spacy.load("en_core_web_sm")

def load_document(path):
    '''
    Load document from given path.
    '''
    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        return f.read()


def find_content_range(text):
    '''
    Find the main content range of an academic document.

    The returned offsets refer to the original document.
    '''

    # Determine the abstract part
    abstract_pattern = re.compile(
        r"(?im)^\s*abstract\s*$"
    )
    abstract_match = abstract_pattern.search(text)
    if abstract_match:
        content_start = abstract_match.end()
        while (
            content_start < len(text)
            and text[content_start].isspace()
        ):
            content_start += 1
    else:
        content_start = 0

    # Determine the reference part
    references_pattern = re.compile(
        r"(?im)^\s*"
        r"(references|bibliography|works\s+cited)"
        r"\s*$"
    )
    references_match = references_pattern.search(
        text,
        pos=content_start,
    )
    if references_match:
        content_end = references_match.start()
    else:
        content_end = len(text)

    return content_start, content_end

def is_reference_marker(text):
    '''
    Check whether a sentence is only a citation/reference marker.
    '''

    text = text.strip()
    normalized = re.sub(r"\s*[.,;:!?]+\s*$", "", text).strip()

    pattern = re.compile(
        r"^(\[\s*\d+(\s*[-,]\s*\d+)*\s*\]|\(\s*\d+(\s*[-,]\s*\d+)*\s*\))$"
    )

    return bool(pattern.match(normalized))


def split_sentences(text):
    '''
    Split document into sentences using spaCy.

    Sentence offsets are always measured against
    the original document.
    '''

    content_start, content_end = find_content_range(text)
    content = text[content_start:content_end]

    doc = nlp(content)
    sentences = []

    for sent in doc.sents:
        raw_text = sent.text
        sentence = raw_text.strip()

        if not sentence:
            continue

        if is_reference_marker(sentence):
            continue

        left_trim = (len(raw_text) - len(raw_text.lstrip()))
        right_trim = (len(raw_text) - len(raw_text.rstrip()))
        offset_start = (content_start + sent.start_char + left_trim)
        offset_end = (content_start + sent.end_char - right_trim)

        sentences.append({
            "text": text[offset_start:offset_end],
            "offset_start": offset_start,
            "offset_end": offset_end,
            "length": offset_end - offset_start,
        })

    return sentences