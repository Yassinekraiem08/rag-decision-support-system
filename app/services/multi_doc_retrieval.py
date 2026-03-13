"""
Multi-Document Retrieval Service

Implements iterative retrieval for queries requiring synthesis across multiple documents.
Standard RAG single-pass retrieval fails on these queries because top-k=3 cannot
cover 4+ required sources. This service detects synthesis queries and runs a
two-pass retrieval to maximize document coverage.

Architecture:
    Standard path:  query → retrieve(k=6) → rerank(k=3) → generate
    Synthesis path: query → retrieve(k=8) → extract_key_terms → retrieve(k=6) →
                    merge_unique_chunks → rerank(k=6) → generate

This is inspired by IRCoT (Interleaved Retrieval with Chain-of-Thought).
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from app.services.pgvector_store import search_chunks_in_db
from app.services.reranker import rerank_chunks

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Query patterns that signal multi-document synthesis need
SYNTHESIS_PATTERNS = [
    r"\ball\b.*\bpapers?\b",
    r"\ball\b.*\bdocuments?\b",
    r"\bcollectively\b",
    r"\bsynthesi[sz]e?\b",
    r"\bacross\b.*\b(papers?|documents?|sources?|corpus)\b",
    r"\bcommon\b.*\b(theme|thread|finding|approach)\b",
    r"\bwhat do.*\b(all|multiple)\b",
    r"\bcompare\b.*\ball\b",
    r"\beverywhere\b",
    r"\bthroughout\b.*\bcorpus\b",
    r"\bmultiple\b.*\bpapers?\b",
    r"\bseveral\b.*\bpapers?\b",
]

MULTI_HOP_PATTERNS = [
    r"\bgiven\b.*(says?|states?|describes?|mentions?)",
    r"\bbased on\b.*\band\b.*\bwhat\b",
    r"\bimplications? of\b.*\bfor\b",
    r"\bhow would\b.*\bif\b",
    r"\brelationship between\b.*\band\b",
    r"\bconnect(ion)?\b.*\bbetween\b",
]


def classify_query(query: str) -> str:
    """
    Classify query retrieval strategy.

    Returns:
        'synthesis'  - requires multi-document synthesis (dynamic top-k + iterative)
        'multi_hop'  - requires chained reasoning (iterative retrieval)
        'standard'   - single document, standard pipeline
    """
    query_lower = query.lower()

    for pattern in SYNTHESIS_PATTERNS:
        if re.search(pattern, query_lower):
            return "synthesis"

    for pattern in MULTI_HOP_PATTERNS:
        if re.search(pattern, query_lower):
            return "multi_hop"

    return "standard"


def extract_key_terms(query: str, chunks: list[tuple[str, str, float]]) -> str:
    """
    Given a query and initial retrieved chunks, extract key entities and concepts
    to use as a second-pass retrieval query.

    This is the core of iterative retrieval: use the LLM to reason about
    what additional information is needed.
    """
    context_preview = "\n\n".join(
        f"[{filename}]: {chunk[:300]}..."
        for chunk, filename, _ in chunks[:3]
    )

    prompt = f"""You are helping improve document retrieval.

Original question: {query}

Initial retrieved context:
{context_preview}

Task: Identify 3-5 specific technical terms, concepts, or named entities from the question
that are NOT yet well-covered by the initial context above. These will be used to
retrieve additional documents.

Return ONLY a comma-separated list of terms. No explanation.
Example output: tactile sensors, manufacturing automation, dexterous manipulation"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": "Extract search terms for document retrieval."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query


def merge_unique_chunks(
    first_pass: list[tuple[str, str, float]],
    second_pass: list[tuple[str, str, float]],
    max_chunks: int = 8
) -> list[tuple[str, str, float]]:
    """
    Merge two retrieval passes, deduplicating by content and maximizing
    document diversity (unique filenames preferred).
    """
    seen_content = set()
    seen_files = set()
    merged = []

    # First pass: add all unique chunks from first retrieval
    for chunk, filename, score in first_pass:
        content_key = chunk[:100]
        if content_key not in seen_content:
            seen_content.add(content_key)
            seen_files.add(filename)
            merged.append((chunk, filename, score))

    # Second pass: prioritize chunks from new documents (diversity)
    new_doc_chunks = []
    existing_doc_chunks = []

    for chunk, filename, score in second_pass:
        content_key = chunk[:100]
        if content_key not in seen_content:
            seen_content.add(content_key)
            if filename not in seen_files:
                new_doc_chunks.append((chunk, filename, score))
                seen_files.add(filename)
            else:
                existing_doc_chunks.append((chunk, filename, score))

    # Add new document chunks first (maximize coverage), then fill remaining slots
    merged.extend(new_doc_chunks)
    remaining_slots = max_chunks - len(merged)
    if remaining_slots > 0:
        merged.extend(existing_doc_chunks[:remaining_slots])

    return merged[:max_chunks]


def retrieve_for_synthesis(query: str, top_k_final: int = 6) -> dict:
    """
    Full iterative retrieval pipeline for synthesis queries.

    Returns dict with chunks and metadata about the retrieval process.
    """
    # Pass 1: broad initial retrieval
    first_pass = search_chunks_in_db(query, top_k=8)
    first_pass_files = list({f for _, f, _ in first_pass})

    # Extract key terms for second pass
    second_query = extract_key_terms(query, first_pass)

    # Pass 2: targeted retrieval using extracted terms
    combined_query = f"{query} {second_query}"
    second_pass = search_chunks_in_db(combined_query, top_k=6)
    second_pass_files = list({f for _, f, _ in second_pass})

    # Merge with diversity preference
    merged = merge_unique_chunks(first_pass, second_pass, max_chunks=8)
    merged_files = list({f for _, f, _ in merged})

    # Rerank merged set, keeping more chunks for synthesis
    reranked = rerank_chunks(query, merged, top_k=top_k_final)
    final_files = list({f for _, f, _ in reranked})

    return {
        "chunks": reranked,
        "metadata": {
            "strategy": "iterative",
            "pass1_docs": first_pass_files,
            "pass2_query": second_query,
            "pass2_docs": second_pass_files,
            "merged_unique_docs": merged_files,
            "final_docs": final_files,
            "doc_coverage": len(final_files)
        }
    }


def smart_retrieve(query: str) -> dict:
    """
    Main entry point. Classifies query and routes to appropriate retrieval strategy.

    Returns:
        {
            "chunks": list of (content, filename, score),
            "strategy": "standard" | "synthesis" | "multi_hop",
            "metadata": dict with retrieval details
        }
    """
    strategy = classify_query(query)

    if strategy == "synthesis":
        result = retrieve_for_synthesis(query, top_k_final=6)
        return {
            "chunks": result["chunks"],
            "strategy": "synthesis",
            "metadata": result["metadata"]
        }
    elif strategy == "multi_hop":
        result = retrieve_for_synthesis(query, top_k_final=5)
        return {
            "chunks": result["chunks"],
            "strategy": "multi_hop",
            "metadata": result["metadata"]
        }
    else:
        # Standard single-pass
        chunks = search_chunks_in_db(query, top_k=6)
        reranked = rerank_chunks(query, chunks, top_k=3)
        return {
            "chunks": reranked,
            "strategy": "standard",
            "metadata": {
                "strategy": "standard",
                "final_docs": list({f for _, f, _ in reranked}),
                "doc_coverage": len({f for _, f, _ in reranked})
            }
        }
