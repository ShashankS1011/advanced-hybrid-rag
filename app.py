import os

# CPU Thread Optimization
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(
    page_title="Production Hybrid RAG Engine",
    layout="wide"
)

st.title("Advanced Production RAG (FlashRank Reranked + Streaming)")
st.caption("Powered by BM25, Chroma Vector Search, FlashRank Cross-Encoder, and Groq Llama 3")

# Sidebar - Document Ingestion & Chunk Inspector
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    st.markdown("---")
    st.markdown("### Active Features")
    st.markdown("""
    * **Parsing:** PyMuPDF (Page-level Metadata)
    * **Embeddings:** `all-MiniLM-L6-v2`
    * **Retrieval:** BM25 + Vector Hybrid Search
    * **Reranker:** `FlashRank` Cross-Encoder
    * **Generation:** Groq Streaming API (`llama-3.1-8b-instant`)
    """)

# Save Uploaded File
pdf_path = None
if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File active: `{uploaded_file.name}`")
elif os.path.exists("data/financial_report.pdf"):
    pdf_path = "data/financial_report.pdf"

# Session State History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource(show_spinner="Indexing and initializing FlashRank reranker...")
def get_pipeline(path: str):
    return RAGPipeline(path)

# Main Application Stream
if pdf_path and os.path.exists(pdf_path):
    try:
        pipeline = get_pipeline(pdf_path)
        
        if user_query := st.chat_input("Ask a question about your document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                # 1. Retrieve & Rerank Chunks
                with st.spinner("Searching and reranking top candidates with FlashRank..."):
                    docs = pipeline.retrieve_chunks(user_query)
                
                # 2. Extract Citations & Page Numbers
                sources_info = []
                for doc in docs:
                    file_name = doc.metadata.get("source_file", "Document")
                    page_num = doc.metadata.get("page_number", "N/A")
                    score = getattr(doc, "relevance_score", None)
                    sources_info.append({
                        "file": file_name,
                        "page": page_num,
                        "score": f"{score:.4f}" if score else "N/A",
                        "text": doc.page_content
                    })

                # 3. Stream Answer in Real-Time
                response_generator = pipeline.stream_answer(user_query, docs)
                full_answer = st.write_stream(response_generator)
                
                # 4. Render Citations & Source Inspector
                pages = sorted(list({s['page'] for s in sources_info if s['page'] != 'N/A'}))
                pages_str = ", ".join([f"Page {p}" for p in pages])
                citation_footer = f"\n\n**Sources:** `{sources_info[0]['file']}` ({pages_str})"
                
                st.markdown(citation_footer)
                
                # 5. Expandable Chunk Inspector (Transparency feature)
                with st.expander("Inspect Re-ranked Search Chunks & FlashRank Scores"):
                    for idx, s in enumerate(sources_info, 1):
                        st.markdown(f"**Chunk #{idx} | Page {s['page']} | Relevance Score: `{s['score']}`**")
                        st.caption(s["text"][:300] + "...")
                        st.divider()

                # Save complete answer into state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer + citation_footer
                })

    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
else:
    st.info("Please upload a PDF in the sidebar to begin.")