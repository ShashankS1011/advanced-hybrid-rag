import logging
import os

# CPU Thread Optimization
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

from src.config import settings
from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Production Hybrid RAG Engine", layout="wide")

st.title("Advanced Production RAG (FlashRank Reranked + Streaming)")
st.caption("Powered by BM25, Chroma Vector Search, FlashRank Cross-Encoder, and Groq Llama 3")

with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    st.markdown("---")
    st.markdown("### Active Features")
    st.markdown(
        """
    * **Parsing:** PyMuPDF (Page-level Metadata)
    * **Embeddings:** `all-MiniLM-L6-v2`
    * **Retrieval:** BM25 + Vector Hybrid Search
    * **Reranker:** `FlashRank` Cross-Encoder
    * **Generation:** Groq Streaming API (`llama-3.1-8b-instant`)
    """
    )


def validate_startup() -> bool:
    try:
        settings.validate_runtime_configuration()
        return True
    except ValueError as exc:
        st.warning(f"Startup check failed: {exc}")
        logger.warning("Startup validation failed: %s", exc)
        return False


pdf_path = None
if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File active: `{uploaded_file.name}`")
elif os.path.exists("data/financial_report.pdf"):
    pdf_path = "data/financial_report.pdf"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.cache_resource(show_spinner="Indexing and initializing FlashRank reranker...")
def get_pipeline(path: str):
    return RAGPipeline(path)


if not validate_startup():
    st.info("Set GROQ_API_KEY and GROQ_MODEL in your environment or .env file to enable generation.")
    st.stop()

if pdf_path and os.path.exists(pdf_path):
    try:
        pipeline = get_pipeline(pdf_path)

        if user_query := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching and reranking top candidates with FlashRank..."):
                        docs = pipeline.retrieve_chunks(user_query)

                    if not docs:
                        no_results = "No relevant chunks were found for that question. Try rephrasing it or improve the document coverage."
                        st.warning(no_results)
                        st.session_state.messages.append({"role": "assistant", "content": no_results})
                        st.stop()

                    sources_info = []
                    for doc in docs:
                        file_name = doc.metadata.get("source_file", "Document")
                        page_num = doc.metadata.get("page_number", "N/A")
                        score = getattr(doc, "relevance_score", None)
                        sources_info.append(
                            {
                                "file": file_name,
                                "page": page_num,
                                "score": f"{score:.4f}" if score else "N/A",
                                "text": doc.page_content,
                            }
                        )

                    response_generator = pipeline.stream_answer(user_query, docs)
                    full_answer = st.write_stream(response_generator)

                    pages = sorted(list({s["page"] for s in sources_info if s["page"] != "N/A"}))
                    pages_str = ", ".join([f"Page {p}" for p in pages])
                    citation_footer = f"\n\n**Sources:** `{sources_info[0]['file']}` ({pages_str})" if sources_info else "\n\n**Sources:** `Document` (N/A)"
                    st.markdown(citation_footer)

                    with st.expander("Inspect Re-ranked Search Chunks & FlashRank Scores"):
                        for idx, s in enumerate(sources_info, 1):
                            st.markdown(f"**Chunk #{idx} | Page {s['page']} | Relevance Score: `{s['score']}`**")
                            st.caption(s["text"][:300] + "...")
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer + citation_footer,
                    })
                except Exception as exc:
                    logger.exception("Search or generation failed")
                    error_text = f"Search or generation failed: {exc}"
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})

    except Exception as exc:
        logger.exception("Error initializing the RAG pipeline")
        st.error(f"Error initializing the RAG pipeline: {exc}")
else:
    st.info("Please upload a PDF in the sidebar to begin.")