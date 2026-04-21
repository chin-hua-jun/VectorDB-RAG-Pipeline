# Vector DB RAG Project

## Overview
Personal exploration project for learning vector databases, embeddings, and RAG (Retrieval-Augmented Generation) using Claude Code documentation as the test dataset.

## Project Structure
- `embedding/` — Document loading, chunking, embedding, and vector store creation
  - `pipeline.py` — Core functions for the embedding pipeline
- `rag/` — RAG chain (not yet implemented)
- `data/` — Cached crawled documents (gitignored)
- `VectorDB.ipynb` — Interactive notebook for experimentation

## Stack
- **Vector DB:** ChromaDB (local)
- **Embeddings:** mxbai-embed-large via Ollama (local, free, GPU-accelerated)
- **LLM (planned):** llama3.1:8b via Ollama
- **Framework:** LangChain
- **Hardware:** RTX 4070 Ti (12GB VRAM)

## Development Workflow
- Python files are maintained by Claude, notebook is for interactive exploration
- The notebook imports from `embedding/` and `rag/` modules
- Crawled docs are cached locally in `data/` to avoid re-crawling during development

## Current State
- Crawled and cached 110 pages of English Claude Code docs from code.claude.com
- Embedding pipeline ready: load → chunk → embed → store in ChromaDB
- Next step: chunk the cached docs, generate embeddings, store in ChromaDB, then build RAG chain

## Commands
- Activate venv: `./venv/Scripts/activate`
- Run Jupyter: `python -m jupyter lab`
- Ollama must be running for embeddings