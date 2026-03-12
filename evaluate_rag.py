import json
from app.services.pgvector_store import search_chunks_in_db
from app.services.generation import generate_answer

TOP_K = 3


def precision_at_k(retrieved, expected_sources):
    hits = 0
    for _, filename, _ in retrieved:
        if filename in expected_sources:
            hits += 1
    return hits / len(retrieved)


def run_evaluation():
    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    total_precision = 0

    print("\nRunning RAG Evaluation\n")

    for sample in dataset:
        question = sample["question"]
        expected = sample["expected_sources"]

        retrieved = search_chunks_in_db(question, top_k=TOP_K)

        precision = precision_at_k(retrieved, expected)
        total_precision += precision

        answer = generate_answer(question, retrieved)

        print("=" * 80)
        print("QUESTION:", question)

        print("\nEXPECTED SOURCES:")
        print(expected)

        print("\nRETRIEVED:")
        for chunk, filename, score in retrieved:
            print(f"{filename} | score={score:.4f}")

        print("\nPRECISION@3:", round(precision, 2))

        print("\nANSWER:")
        print(answer)
        print()

    avg_precision = total_precision / len(dataset)

    print("=" * 80)
    print("AVERAGE PRECISION@3:", round(avg_precision, 3))


if __name__ == "__main__":
    run_evaluation()