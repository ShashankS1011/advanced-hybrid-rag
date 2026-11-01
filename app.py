import datetime
import logging
import os
import sys
import time
from pathlib import Path

# Workaround for ChromaDB requiring SQLite >= 3.35.0 on Linux (Streamlit Community Cloud)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# CPU Thread Optimization
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import find_dotenv, load_dotenv

BASE_DIR = Path(__file__).resolve().parent
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, override=True)
else:
    load_dotenv(find_dotenv(), override=True)

import streamlit as st

from src.config import settings
from src.ingest import compute_file_hash

st.set_page_config(
    page_title="Production Hybrid RAG Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_api_key() -> tuple[str, str]:
    """Resolves Groq API key and source across session state, environment/.env, and Streamlit secrets."""
    # 1. User manual input stored in session state
    if st.session_state.get("user_groq_key"):
        val = str(st.session_state["user_groq_key"]).strip()
        if val:
            return val, "User Session"

    # 2. Settings / Environment variable / .env file
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
        return settings.GROQ_API_KEY.strip(), "Environment / .env"
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key, "Environment / .env"

    # 3. Streamlit Cloud Secrets (exhaustive case-insensitive search across root keys and sub-sections)
    try:
        if hasattr(st, "secrets"):
            possible_keys = [
                "GROQ_API_KEY",
                "groq_api_key",
                "Groq_Api_Key",
                "GROQ_KEY",
                "groq_key",
                "API_KEY",
                "api_key",
            ]
            for k in possible_keys:
                try:
                    if k in st.secrets and str(st.secrets[k]).strip():
                        return str(st.secrets[k]).strip(), "Streamlit Secrets"
                except Exception:
                    pass

            for sec_name in st.secrets:
                try:
                    sec = st.secrets[sec_name]
                    if hasattr(sec, "get") or isinstance(sec, dict):
                        for k in possible_keys:
                            if k in sec and str(sec[k]).strip():
                                return str(sec[k]).strip(), "Streamlit Secrets"
                except Exception:
                    pass
    except Exception:
        pass

    return "", ""


# Synchronize active API key
active_key, key_source = resolve_api_key()
if active_key:
    settings.GROQ_API_KEY = active_key
    os.environ["GROQ_API_KEY"] = active_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource
def get_daily_usage_tracker():
    return {"date": str(datetime.date.today()), "count": 0}


usage_tracker = get_daily_usage_tracker()
today_str = str(datetime.date.today())
if usage_tracker.get("date") != today_str:
    usage_tracker["date"] = today_str
    usage_tracker["count"] = 0

default_pdf = BASE_DIR / "data" / "financial_report.pdf"

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("Control Panel")

    st.subheader("API Configuration")
    if active_key:
        masked = f"{active_key[:6]}...{active_key[-4:]}" if len(active_key) > 12 else "Configured"
        st.success(f"API Key Active: {masked}")
        st.caption(f"Source: {key_source}")
        if key_source != "User Session":
            remaining = max(0, settings.DAILY_GLOBAL_QUERY_LIMIT - usage_tracker["count"])
            st.caption(f"Shared Demo Quota: {remaining}/{settings.DAILY_GLOBAL_QUERY_LIMIT} queries remaining today")

        with st.expander("Custom Key Override (Optional)"):
            st.caption(
                "A system key is already active. You can supply your own Groq key below to bypass shared quotas."
            )
            entered_key = st.text_input(
                "Groq API Key Override",
                value=st.session_state.get("user_groq_key", ""),
                type="password",
                placeholder="gsk_...",
                key="api_key_override_input",
            )
            if entered_key and entered_key.strip() != st.session_state.get("user_groq_key", ""):
                st.session_state["user_groq_key"] = entered_key.strip()
                settings.GROQ_API_KEY = entered_key.strip()
                os.environ["GROQ_API_KEY"] = entered_key.strip()
                st.rerun()
            elif not entered_key and st.session_state.get("user_groq_key"):
                st.session_state.pop("user_groq_key", None)
                st.rerun()
    else:
        st.warning("No API Key detected.")
        st.markdown(
            "**To activate:**\n"
            "* Enter your free Groq API key below for this session, **or**\n"
            '* Add `GROQ_API_KEY = "gsk_..."` in **Streamlit Cloud > App Settings > Secrets**.'
        )
        entered_key = st.text_input(
            "Groq API Key",
            value=st.session_state.get("user_groq_key", ""),
            type="password",
            placeholder="gsk_...",
            help="Free Groq API key from https://console.groq.com",
            key="api_key_input",
        )
        if entered_key and entered_key.strip() != st.session_state.get("user_groq_key", ""):
            st.session_state["user_groq_key"] = entered_key.strip()
            settings.GROQ_API_KEY = entered_key.strip()
            os.environ["GROQ_API_KEY"] = entered_key.strip()
            st.rerun()

    st.markdown("---")
    st.subheader("Document Management")
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])

    st.markdown("---")
    st.subheader("Retrieval Tuning")
    rerank_top_n = st.slider(
        "Rerank Top Chunks",
        min_value=1,
        max_value=10,
        value=settings.RERANK_TOP_N,
        help="Number of top candidates passed to FlashRank cross-encoder before context injection.",
    )
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.TEMPERATURE,
        step=0.05,
        help="Sampling temperature for LLM generation. Lower values provide more deterministic answers.",
    )
    settings.RERANK_TOP_N = rerank_top_n
    settings.TEMPERATURE = temperature

    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("Architecture Details")
    st.markdown(
        """
    * **Ingestion:** PyMuPDF + Markdown Table Parsing
    * **Embeddings:** all-MiniLM-L6-v2 (CPU Optimized)
    * **Hybrid Retrieval:** BM25 (Lexical) + Chroma (Vector)
    * **Reranking:** FlashRank ms-marco-TinyBERT
    * **Generation:** Groq Streaming (llama-3.1-8b-instant)
    """
    )


# --- Document Ingestion & Resolution ---
pdf_path = None
doc_hash = ""
active_doc_name = ""

if uploaded_file is not None:
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    target_file = data_dir / safe_name
    with open(target_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = str(target_file)
    doc_hash = compute_file_hash(pdf_path)
    active_doc_name = safe_name
    st.sidebar.info(f"Loaded: {safe_name} (ID: {doc_hash})")
elif default_pdf.exists():
    pdf_path = str(default_pdf)
    doc_hash = compute_file_hash(pdf_path)
    active_doc_name = "financial_report.pdf"
    st.sidebar.info("Preloaded: financial_report.pdf (Acme Corp Q4)")


@st.cache_resource(show_spinner="Indexing document and initializing FlashRank cross-encoder...")
def get_pipeline(path: str, content_hash: str):
    from src.pipeline import RAGPipeline

    return RAGPipeline(path)


# --- Main Application Header ---
st.title("Advanced Production Hybrid RAG Engine")
st.caption(
    "High-precision document Q&A combining BM25 keyword search, Chroma vector embeddings, FlashRank reranking, and Groq streaming."
)

if pdf_path and os.path.exists(pdf_path):
    try:
        pipeline = get_pipeline(pdf_path, doc_hash)
        doc_pages = pipeline.metrics.get("document_pages", 1)
        doc_chunks = pipeline.metrics.get("chunks_created", 1)

        # Overview Metrics Banner
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Document", active_doc_name)
        m2.metric("Document Pages", doc_pages)
        m3.metric("Indexed Chunks", doc_chunks)
        m4.metric("Reranked Top-N", settings.RERANK_TOP_N)

        st.markdown("---")

        # Suggested Questions for Demo Document
        if active_doc_name == "financial_report.pdf":
            st.markdown("##### Quick Suggested Questions")
            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1:
                if st.button("Q4 2025 Net Revenue", use_container_width=True):
                    st.session_state["pending_query"] = (
                        "What was the total revenue recorded by Acme Corporation in Q4 2025?"
                    )
            with qcol2:
                if st.button("EBITDA & Cash Reserves", use_container_width=True):
                    st.session_state["pending_query"] = (
                        "What were the operating expenses, EBITDA margin, and cash reserves in Q4 2025?"
                    )
            with qcol3:
                if st.button("Operational Risks for FY2026", use_container_width=True):
                    st.session_state["pending_query"] = "What are the primary operational risks identified for FY2026?"

        # Chat History Rendering
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Determine if a prompt was triggered via button or input
        user_query = None
        if st.session_state.get("pending_query"):
            user_query = st.session_state.pop("pending_query")
        else:
            chat_input_text = st.chat_input("Ask a question about your document...")
            if chat_input_text:
                user_query = chat_input_text

        if user_query:
            # Check API Key availability
            if not active_key:
                st.warning(
                    "Please configure your Groq API Key in the sidebar or configure GROQ_API_KEY in Streamlit Cloud Secrets to run queries."
                )
                st.stop()

            # Safety Guardrail 1: Question length
            if len(user_query.strip()) > settings.MAX_QUERY_LENGTH:
                st.error(
                    f"Question exceeds the limit of {settings.MAX_QUERY_LENGTH} characters. Please submit a shorter question."
                )
                st.stop()

            # Safety Guardrail 2: Request cooldown
            now = time.time()
            last_time = st.session_state.get("last_query_time", 0.0)
            if (now - last_time) < settings.RATE_LIMIT_SECONDS:
                st.warning(f"Rate limit active. Please wait {settings.RATE_LIMIT_SECONDS} seconds between requests.")
                st.stop()

            # Safety Guardrail 3: Cost and Quota protection for shared key
            is_personal = bool(st.session_state.get("user_groq_key"))
            if not is_personal:
                if usage_tracker["count"] >= settings.DAILY_GLOBAL_QUERY_LIMIT:
                    st.error(
                        "Daily demo quota reached for this shared instance. "
                        "To continue querying without limits, please provide your own free Groq API key in the sidebar."
                    )
                    st.stop()

                session_count = st.session_state.get("session_queries", 0)
                if session_count >= settings.MAX_SESSION_QUERIES:
                    st.warning(
                        f"Session limit of {settings.MAX_SESSION_QUERIES} queries reached for this demo. "
                        "Please enter your own free Groq API key in the sidebar to continue querying."
                    )
                    st.stop()

            st.session_state.last_query_time = now
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Retrieving candidates and reranking with FlashRank..."):
                        docs = pipeline.retrieve_chunks(user_query, st.session_state.messages[:-1])

                    if not docs:
                        no_results = "No relevant chunks were found for that question. Try rephrasing it or adjust retrieval settings."
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

                    if not is_personal:
                        usage_tracker["count"] += 1
                        st.session_state.session_queries = st.session_state.get("session_queries", 0) + 1

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

                    with st.expander("Inspect Re-ranked Search Chunks & FlashRank Relevance Scores"):
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
    st.info("Please upload a PDF document in the sidebar to begin.")
