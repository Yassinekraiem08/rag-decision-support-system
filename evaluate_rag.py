from app.services.pgvector_store import search_chunks_in_db
from app.services.generation import generate_answer

TEST_QUERIES = [
    "What is embodied AI?",
    "What is Retrieval-Augmented Generation?",
    "Why is embodied intelligence important?",
    "What do the indexed papers say about robotics?"
]


def evaluate():
    print("\nRunning RAG evaluation...\n")

    for question in TEST_QUERIES:
        retrieved = search_chunks_in_db(question, top_k=3)
        answer = generate_answer(question, retrieved)

        print("=" * 80)
        print("QUESTION:")
        print(question)

        print("\nTOP RETRIEVED SOURCES:")
        for chunk, filename, score in retrieved:
            print(f"- {filename} | score={score:.4f}")

        print("\nANSWER:")
        print(answer)
        print()


if __name__ == "__main__":
    evaluate()