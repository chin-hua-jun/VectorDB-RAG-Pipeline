# VectorDB RAG Pipeline

> Personal exploration project for learning vector databases, embeddings, and RAG (Retrieval-Augmented Generation).
> Pipeline works for any documents, for this example I used Claude Code documents as it's something I could see myself using a RAG-augmented local LLM for checking questions.

## Current Progress

**Stage 1 — Vector Database: Complete ✅**

- Crawled and cached 114 pages of English Claude Code documentation from [code.claude.com](https://code.claude.com)
- Extracts main content only (targets `div#content`, skipping nav/sidebar/footer)
- Cleans and splits documents into 2,572 chunks (~943 chars avg, 200 char overlap)
- Generates embeddings using `mxbai-embed-large` via Ollama (local, GPU-accelerated)
- Stores vectors in a local ChromaDB instance

**Stage 2 — RAG Chain: In Progress 🔄**

- Retriever and prompt template defined in `rag/chain.py`
- LLM integration and full chain wiring not yet complete

**Stage 3 — Evaluation: Not started**

---

## Full Plan

### Stage 1 — Document Ingestion & Vector Store

1. **Crawl** — Use LangChain's `SitemapLoader` to pull all English Claude Code docs from the sitemap
2. **Extract** — Parse only the main content element from each page (no nav, sidebar, or footer)
3. **Clean** — Strip short/noisy lines and collapse blank runs
4. **Chunk** — Split with `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200)
5. **Embed** — Generate vector embeddings using `mxbai-embed-large` via Ollama
6. **Store** — Persist in ChromaDB for semantic similarity search

### Stage 2 — RAG Chain

1. **Retriever** — Query ChromaDB for the top-k most relevant chunks
2. **Prompt** — Inject retrieved context + user question into a structured prompt template
3. **LLM** — Pass to `llama3.1:8b` via Ollama for answer generation
4. **Chain** — Wire everything together using LangChain Expression Language (LCEL): `retriever | prompt | llm | output_parser`

### Stage 3 — Evaluation

1. **Test questions** — Build a small set of ground-truth Q&A pairs from the docs
2. **Retrieval quality** — Measure whether the right chunks are being retrieved (recall@k)
3. **Answer quality** — Assess faithfulness and relevance of generated answers
4. **Iterate** — Tune chunk size, overlap, retrieval k, and prompt as needed

---

## Stack

| Component     | Choice                                      |
|---------------|---------------------------------------------|
| Vector DB     | [ChromaDB](https://www.trychroma.com/) (local) |
| Embeddings    | `mxbai-embed-large` via [Ollama](https://ollama.com/) |
| LLM           | `llama3.1:8b` via Ollama *(planned)*        |
| Framework     | [LangChain](https://www.langchain.com/)     |
| Notebook      | JupyterLab                                  |
| Hardware      | RTX 4070 Ti (12GB VRAM)                     |

---

## Project Structure

```
.
├── embedding/
│   └── pipeline.py       # Crawl, clean, chunk, embed, store
├── rag/
│   └── chain.py          # Retriever, prompt template, LLM chain
├── VectorDB.ipynb         # Interactive notebook for exploration
├── data/                  # Cached crawled docs (gitignored)
└── chroma_db/             # Local vector store (gitignored)
```

---

## Setup

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com/) running locally with `mxbai-embed-large` and `llama3.1:8b` pulled.

```bash
# Create and activate virtual environment
python -m venv venv
./venv/Scripts/activate   # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull mxbai-embed-large
ollama pull llama3.1:8b
```

### Run the embedding pipeline

```bash
python -m embedding.pipeline
```

This loads cached docs, cleans and chunks them, generates embeddings, and writes the ChromaDB vector store to `./chroma_db`.

To re-crawl from scratch, call `load_claude_docs()` + `save_docs()` from `embedding/pipeline.py` first.
