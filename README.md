# Advanced Production Hybrid RAG Engine

A high-performance, low-latency Retrieval-Augmented Generation (RAG) system engineered for complex document understanding. This system combines **Sparse Search (BM25)** and **Dense Vector Search (ChromaDB)** with **FlashRank Cross-Encoder Re-ranking** and **Groq Llama 3 Streaming Inference** to deliver highly accurate answers accompanied by precise, page-level source citations.

---

## Architecture Overview

* **PDF Document**
* **PyMuPDF Parser** (Page-Level Metadata Tagging)
* **BM25 Retriever** (Keyword Matching) + **Chroma Vector Database** (`all-MiniLM-L6-v2` Embeddings)
* **Candidate Chunks (Top-K)**
* **FlashRank Cross-Encoder** (Neural Re-ranking)
* **High-Relevance Chunks + Prompt**
* **Groq LLM Inference** (`llama-3.1-8b-instant`)
* **Streamlit UI with Streaming** (Live Tokens + Page Citations + Chunk Inspector)

### Core Features

* **Hybrid Retrieval (Dense + Sparse):** Combines lexical search via BM25 and semantic vector search via ChromaDB for high recall across structured data and domain-specific vocabulary.
* **FlashRank Cross-Encoder Re-ranking:** Re-scores and filters retrieved chunks using ultra-fast cross-encoders to eliminate noise and reduce context window overload.
* **Page-Level Metadata & Citations:** Tracks exact page numbers throughout chunking, retrieval, and generation to provide verifiable inline source citations.
* **Real-time Token Streaming:** Integrates Groq's low-latency API (`llama-3.1-8b-instant`) to stream responses back to the user instantly.
* **Chunk Inspection & Transparency:** Includes a dedicated UI inspector showing raw retrieved chunks, page numbers, and FlashRank relevance scores.

---

## Repository Structure

```text
.
├── data/                  # Input PDF directory (e.g., financial_report.pdf)
├── src/
│   ├── __init__.py
│   ├── ingester.py        # PyMuPDF parser & text chunker
│   ├── retriever.py       # BM25 + Vector Hybrid search & FlashRank setup
│   └── pipeline.py        # End-to-end RAG orchestrator & Groq streaming
├── app.py                 # Streamlit web application frontend
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # Project documentation
```

---

## Quickstart & Installation

### Prerequisites

* Python 3.10+
* A Groq API key

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/advanced-hybrid-rag.git](https://github.com/your-username/advanced-hybrid-rag.git)
cd advanced-hybrid-rag
```

### 2. Set Up Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

---

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Usage Guide

1. **Upload Document:** Use the sidebar to upload any PDF (e.g., financial reports, legal contracts, research papers).
2. **Ask Questions:** Query your document using natural language in the chat bar.
3. **Review Citations:** Check the inline page references attached to the response.
4. **Inspect Chunks:** Expand the **Inspect Re-ranked Search Chunks & FlashRank Scores** section to evaluate retrieved text segments and relevance scores.

---

## Tech Stack

* **Frontend:** Streamlit
* **Document Parsing:** PyMuPDF (`fitz`)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** ChromaDB
* **Keyword Search:** `rank_bm25`
* **Reranker:** `FlashRank`
* **LLM Engine:** Groq API (`llama-3.1-8b-instant`)
* **Framework:** LangChain ecosystem

---

## License

Distributed under the MIT License. See `LICENSE` for more information.