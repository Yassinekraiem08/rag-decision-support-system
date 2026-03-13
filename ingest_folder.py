import os
import re
from app.utils.loaders import load_text_file, load_pdf_file
from app.services.chunking import chunk_text
from app.services.pgvector_store import store_document_chunks, clear_database

DOCS_FOLDER = "docs"


def classify_domain(filename: str) -> str:
    """Classify a document as 'literary' (Gutenberg) or 'technical' based on filename."""
    if re.match(r"pg\d+\.txt", filename):
        return "literary"
    return "technical"


def ingest_folder(folder_path: str):
    total_docs = 0
    total_chunks = 0

    print(f"Scanning folder: {folder_path}")
    clear_database()

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                print(f"\nProcessing {filename}")

                if filename.endswith(".txt"):
                    text = load_text_file(file_path)
                elif filename.endswith(".pdf"):
                    text = load_pdf_file(file_path)
                else:
                    print(f"Skipping unsupported file: {filename}")
                    continue

                chunks = chunk_text(text, chunk_size=500, overlap=50)
                domain = classify_domain(filename)
                print(f"{filename} produced {len(chunks)} chunks (domain={domain})")

                store_document_chunks(filename, chunks, domain=domain)

                total_docs += 1
                total_chunks += len(chunks)

                print(f"Ingested {filename} -> {len(chunks)} chunks")

            except Exception as e:
                print(f"Failed to ingest {filename}: {e}")

    print(f"\nDone. Indexed {total_docs} documents and {total_chunks} chunks.")


if __name__ == "__main__":
    ingest_folder(DOCS_FOLDER)