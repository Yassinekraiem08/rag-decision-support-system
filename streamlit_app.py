import time
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)


def render_verification_badge(verdict: str):
    verdict = verdict.upper().strip()

    if verdict == "SUPPORTED":
        st.success(f"Verification: {verdict}")
    elif verdict == "PARTIALLY_SUPPORTED":
        st.warning(f"Verification: {verdict}")
    elif verdict == "UNSUPPORTED":
        st.error(f"Verification: {verdict}")
    else:
        st.info(f"Verification: {verdict}")


st.title("🧠 RAG Knowledge Assistant")
st.caption("Production-style Retrieval-Augmented Generation system with groundedness verification")

tab1, tab2 = st.tabs(["Ask Questions", "Upload Documents"])

with tab2:
    st.subheader("Upload a document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Uploading and indexing document..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                try:
                    response = requests.post(f"{API_URL}/ingest", files=files, timeout=120)

                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"Indexed {data['filename']} successfully with {data['chunks_stored']} chunks."
                        )
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.RequestException as e:
                    st.error(f"Could not connect to backend: {e}")

with tab1:
    st.subheader("Ask a question")
    question = st.text_input(
        "Enter your question",
        placeholder="Example: What is embodied AI?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("Ask")
    with col2:
        show_chunks = st.checkbox("Show retrieved chunks", value=True)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            start_time = time.time()

            with st.spinner("Retrieving, generating, and verifying answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question},
                        timeout=120
                    )
                    elapsed = time.time() - start_time

                    if response.status_code != 200:
                        st.error(f"Query failed: {response.text}")
                    else:
                        data = response.json()

                        answer = data.get("answer", "")
                        references = data.get("references", [])
                        confidence = data.get("confidence", 0.0)
                        confidence_reasoning = data.get("confidence_reasoning", "")
                        verification = data.get("verification", {})
                        retrieved_chunks = data.get("retrieved_chunks", [])

                        verdict = verification.get("verdict", "UNKNOWN")
                        reason = verification.get("reason", "No verification reason provided.")

                        # Display answer
                        st.markdown("## Answer")
                        st.write(answer)

                        # Display references
                        if references:
                            st.markdown("### References")
                            for ref in references:
                                st.caption(ref)

                        # Display confidence score
                        st.markdown("## Confidence")

                        # Color-code confidence
                        if confidence >= 0.8:
                            st.success(f"**{confidence:.2f}** - {confidence_reasoning}")
                        elif confidence >= 0.6:
                            st.info(f"**{confidence:.2f}** - {confidence_reasoning}")
                        else:
                            st.warning(f"**{confidence:.2f}** - {confidence_reasoning}")

                        # Display verification
                        st.markdown("## Verification")
                        render_verification_badge(verdict)
                        st.write(reason)

                        st.caption(f"Response time: {elapsed:.2f} seconds")

                        if show_chunks:
                            st.markdown("## Retrieved Chunks")

                            if not retrieved_chunks:
                                st.info("No retrieved chunks returned.")
                            else:
                                for i, chunk in enumerate(retrieved_chunks, start=1):
                                    filename = chunk.get("filename", "Unknown file")
                                    score = chunk.get("score", 0.0)
                                    content = chunk.get("content", "")

                                    with st.expander(
                                        f"{i}. {filename} — score: {score:.4f}",
                                        expanded=(i == 1)
                                    ):
                                        st.write(content)

                except requests.RequestException as e:
                    st.error(f"Could not connect to backend: {e}")