import os
from xml.sax.saxutils import escape

def save_predictions_text(
    predictions,
    suspicious_text,
    source_text,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, prediction in enumerate(predictions, start=1):
            suspicious_start = prediction["suspicious_start"]
            suspicious_length = prediction["suspicious_length"]
            source_start = prediction["source_start"]
            source_length = prediction["source_length"]

            suspicious_end = suspicious_start + suspicious_length
            source_end = source_start + source_length

            suspicious = suspicious_text[suspicious_start:suspicious_end]
            source = source_text[source_start:source_end]

            f.write(f"PREDICTION {idx}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Suspicious: {suspicious_start} - {suspicious_end}\n")
            f.write(f"Source: {source_start} - {source_end}\n\n")
            f.write("SUSPICIOUS\n")
            f.write(suspicious.strip() + "\n\n")
            f.write("SOURCE\n")
            f.write(source.strip() + "\n\n")
            f.write("=" * 80 + "\n\n")


def save_predictions_xml(
    predictions,
    output_path,
    suspicious_filename,
    source_filename,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            f'<document reference="{escape(suspicious_filename)}">\n'
        )

        for prediction in predictions:
            f.write(
                '  <feature '
                'name="plagiarism" '
                'type="prediction" '
                f'this_offset="{prediction["suspicious_start"]}" '
                f'this_length="{prediction["suspicious_length"]}" '
                f'source_reference="{escape(source_filename)}" '
                f'source_offset="{prediction["source_start"]}" '
                f'source_length="{prediction["source_length"]}"'
                '/>\n'
            )

        f.write("</document>\n")