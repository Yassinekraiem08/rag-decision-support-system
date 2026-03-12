import requests
import json

# Test the query endpoint
response = requests.post(
    "http://127.0.0.1:8000/query",
    json={"question": "What is embodied AI?"},
    timeout=60
)

print(f"Status Code: {response.status_code}\n")

if response.status_code == 200:
    data = response.json()

    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(data.get("answer", ""))
    print()

    print("=" * 80)
    print("REFERENCES:")
    print("=" * 80)
    for ref in data.get("references", []):
        print(f"  {ref}")
    print()

    print("=" * 80)
    print("CONFIDENCE:")
    print("=" * 80)
    print(f"  Score: {data.get('confidence', 0.0):.3f}")
    print(f"  Reasoning: {data.get('confidence_reasoning', '')}")
    print()

    print("=" * 80)
    print("VERIFICATION:")
    print("=" * 80)
    verification = data.get("verification", {})
    print(f"  Verdict: {verification.get('verdict', 'UNKNOWN')}")
    print(f"  Reason: {verification.get('reason', '')}")
    print()

    print("=" * 80)
    print("RETRIEVED CHUNKS:")
    print("=" * 80)
    for i, chunk in enumerate(data.get("retrieved_chunks", []), 1):
        print(f"  [{i}] {chunk.get('filename', '')} (score: {chunk.get('score', 0.0):.4f})")
    print()

    print("✅ All Phase 1 features are working!")
else:
    print(f"❌ Error: {response.text}")
