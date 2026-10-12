# Advanced Production Hybrid RAG Engine

This project is a retrieval-augmented generation application designed to answer questions from PDF documents using a hybrid search pipeline. It combines lexical retrieval, semantic retrieval, reranking, and large-language-model generation to provide grounded answers with page-level source references.

The primary purpose is to enable accurate, explainable document Q&A over extended PDFs such as financial reports, contracts, policy documents, and technical documentation. The system is built for practical use in document analysis workflows where answer quality, traceability, and transparency matter.

## What This Project Does

The application ingests a PDF, splits it into manageable text chunks, retrieves the most relevant passages using both keyword and semantic search, reranks the best candidates, and then uses a Groq-hosted language model to answer the user question using only the retrieved context.

This workflow helps reduce hallucination risk by constraining generation to the actual document content. It also surfaces source page references so users can verify the answer against the underlying document.

## Core Components

- **Document ingestion:** Reads PDF files and extracts text while preserving page metadata.
- **Chunking:** Splits documents into overlapping text blocks for targeted retrieval.
- **Hybrid retrieval:** Uses BM25 for keyword matching and Chroma for semantic similarity search.
- **Reranking:** Reorders retrieved chunks to prioritize the most relevant context.
- **Generation:** Uses Groq LLM inference to produce responses grounded in the retrieved content.
- **User experience:** Exposes the workflow through a Streamlit interface with visible citations and chunk inspection.

## Repository Structure

```text
.
├── data/                  # Input PDF directory
├── src/
│   ├── __init__.py
│   ├── config.py          # Environment and runtime settings
│   ├── ingest.py          # PDF parsing and chunking logic
│   ├── retriever.py       # Hybrid retrieval and reranking logic
│   └── pipeline.py        # End-to-end pipeline orchestration
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── tests/                 # Automated tests
├── README.md              # Project documentation
├── .github/workflows/     # CI pipeline configuration
└── .gitignore             # Repository exclusions
```

## Prerequisites

- Python 3.11+
- A Groq API key
- Access to the required Python dependencies listed in the project requirements

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShashankS1011/advanced-hybrid-rag.git
cd advanced-hybrid-rag
```

### 2. Create a virtual environment

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Optional values can also be configured for retrieval tuning and storage:

```env
CHROMA_DB_DIR=./chroma_db
CHROMA_COLLECTION_NAME=advanced_hybrid_rag
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DENSE_K=12
SPARSE_K=12
RERANK_TOP_N=4
```

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, typically:

```text
http://localhost:8501
```

## How to Use It

- Upload a PDF document from the sidebar.
- Ask a question about the document in the chat input.
- Review the generated answer and the page citations shown in the output.
- Inspect the retrieved chunks to understand which sections influenced the final response.

## Development and Validation

Run the automated tests:

```bash
python -m pytest -q
```

The project includes CI via GitHub Actions to validate the codebase on pushes and pull requests.

## Technology Stack

- **Frontend:** Streamlit
- **PDF parsing:** PyMuPDF
- **Embeddings:** sentence-transformers
- **Vector search:** Chroma
- **Keyword search:** BM25
- **Reranking:** FlashRank
- **LLM inference:** Groq
- **Framework:** LangChain

## Summary

This project provides a practical hybrid RAG workflow for document-grounded question answering. It is designed to help users extract precise answers from large PDFs in a way that is explainable, traceable, and easy to interact with from a browser interface.