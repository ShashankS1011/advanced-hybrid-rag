import logging
import os
from pathlib import Path

# CPU Thread Optimization
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

from src.config import settings
from src.ingest import compute_file_hash
from src.pipeline import RAGPipeline

# Synchronize Streamlit Cloud secrets if available
if not settings.GROQ_API_KEY:
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            settings.GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"]).strip()
            os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Production Hybrid RAG Engine", layout="wide")

st.title("Advanced Production RAG (FlashRank Reranked + Streaming)")
st.caption("Powered by BM25, Chroma Vector Search, FlashRank Cross-Encoder, and Groq Llama 3")

with st.sidebar:
    st.header("API Configuration")
    if not settings.GROQ_API_KEY:
        api_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free Groq API key from https://console.groq.com",
        )
        if api_input:
            settings.GROQ_API_KEY = api_input.strip()
            os.environ["GROQ_API_KEY"] = api_input.strip()
            st.rerun()
    else:
        st.success("Groq API Key configured")

    st.markdown("---")
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    st.markdown("---")
    st.header("Retrieval Settings")
    rerank_top_n = st.slider(
        "Rerank Top N Chunks",
        min_value=1,
        max_value=10,
        value=settings.RERANK_TOP_N,
    )
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.TEMPERATURE,
        step=0.05,
    )
    settings.RERANK_TOP_N = rerank_top_n
    settings.TEMPERATURE = temperature

    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Active Features")
    st.markdown(
        """
    * **Parsing:** PyMuPDF + Table Detection
    * **Embeddings:** `all-MiniLM-L6-v2`
    * **Retrieval:** BM25 + Vector Hybrid Search
    * **Reranker:** `FlashRank` Cross-Encoder
    * **Memory:** Conversational Query Reformulation
    * **Generation:** Groq Streaming API (`llama-3.1-8b-instant`)
    """
    )


def validate_startup() -> bool:
    if not settings.GROQ_API_KEY:
        st.info("Enter your free **Groq API Key** in the sidebar (or configure secrets) to begin chatting.")
        return False
    try:
        settings.validate_runtime_configuration()
        return True
    except ValueError as exc:
        st.warning(f"Startup check failed: {exc}")
        logger.warning("Startup validation failed: %s", exc)
        return False


pdf_path = None
doc_hash = ""

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    # Sanitize uploaded filename to prevent directory traversal
    safe_name = Path(uploaded_file.name).name
    pdf_path = os.path.join("data", safe_name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    doc_hash = compute_file_hash(pdf_path)
    st.sidebar.success(f"Active file: `{safe_name}` (ID: `{doc_hash}`)")
elif os.path.exists("data/financial_report.pdf"):
    pdf_path = "data/financial_report.pdf"
    doc_hash = compute_file_hash(pdf_path)
    st.sidebar.info("Preloaded document: `financial_report.pdf` active")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.cache_resource(show_spinner="Indexing document and initializing FlashRank reranker...")
def get_pipeline(path: str, content_hash: str):
    return RAGPipeline(path)


if not validate_startup():
    st.stop()

if pdf_path and os.path.exists(pdf_path):
    try:
        pipeline = get_pipeline(pdf_path, doc_hash)
        st.sidebar.info(
            f"**Pages:** {pipeline.metrics.get('document_pages', 1)} | "
            f"**Chunks:** {pipeline.metrics.get('chunks_created', 1)}"
        )

        if user_query := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching and reranking top candidates with FlashRank..."):
                        docs = pipeline.retrieve_chunks(user_query, st.session_state.messages[:-1])

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
                                "score": f"{score:.4f}" if score is not None else "N/A",
                                "text": doc.page_content,
                            }
                        )

                    response_generator = pipeline.stream_answer(
                        user_query,
                        docs,
                        st.session_state.messages[:-1],
                    )
                    full_answer = st.write_stream(response_generator)

                    pages = sorted({s["page"] for s in sources_info if s["page"] != "N/A"})
                    pages_str = ", ".join([f"Page {p}" for p in pages])
                    source_files = sorted({s["file"] for s in sources_info})
                    files_str = ", ".join([f"`{f}`" for f in source_files])
                    citation_footer = (
                        f"\n\n**Sources:** {files_str} ({pages_str})"
                        if sources_info
                        else "\n\n**Sources:** `Document` (N/A)"
                    )
                    st.markdown(citation_footer)

                    with st.expander("Inspect Re-ranked Search Chunks & FlashRank Scores"):
                        for idx, s in enumerate(sources_info, 1):
                            st.markdown(
                                f"**Chunk #{idx} | {s['file']} | Page {s['page']} | Relevance Score: `{s['score']}`**"
                            )
                            st.caption(s["text"][:300] + "...")
                            st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_answer + citation_footer,
                        }
                    )
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
