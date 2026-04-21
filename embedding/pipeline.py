import os
import json
from pathlib import Path
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

os.environ.setdefault("USER_AGENT", "vector-db-rag-crawler/1.0")

DATA_DIR = Path(__file__).parent.parent / "data"


# --- Document Loading ---

def _extract_main_content(soup):
    main = soup.find("div", id="content")
    if main:
        return main.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def load_from_sitemap(web_path, filter_urls=None):
    """Load documents from a sitemap URL, optionally filtering by URL patterns."""
    loader = SitemapLoader(
        web_path=web_path,
        filter_urls=filter_urls,
        parsing_function=_extract_main_content,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from {web_path}")
    return docs


def load_claude_docs():
    """Load all English Claude Code documentation."""
    return load_from_sitemap(
        web_path="https://code.claude.com/sitemap.xml",
        filter_urls=["https://code.claude.com/docs/en/"],
    )


# --- Cache (local disk) ---

def save_docs(docs, name):
    """Save documents to disk as JSON for reuse without re-crawling."""
    path = DATA_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], f, indent=2)
    print(f"Saved {len(docs)} docs to {path}")

def load_saved_docs(name):
    """Load previously saved documents from disk."""
    path = DATA_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    print(f"Loaded {len(docs)} docs from {path}")
    return docs

def clean_docs(docs):
    """Remove noise from crawled documents: short lines and blank runs."""
    cleaned = []
    for doc in docs:
        lines = doc.page_content.splitlines()

        # Drop very short lines (stray punctuation, single words from UI elements)
        lines = [l for l in lines if len(l.strip()) > 3]

        # Collapse runs of consecutive blank lines into one
        deduped = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            deduped.append(line)
            prev_blank = is_blank

        content = "\n".join(deduped).strip()
        if content:
            cleaned.append(Document(page_content=content, metadata=doc.metadata))

    print(f"Cleaned {len(docs)} docs -> {len(cleaned)} non-empty docs")
    return cleaned


# --- Chunking ---

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} docs into {len(chunks)} chunks")
    return chunks

# --- Embeddings ---
def get_ollama_embeddings(model="mxbai-embed-large"):
    """Get embeddings via Ollama (local, free)."""
    return OllamaEmbeddings(model=model)

# --- Vector Store ---

def create_vectorstore(chunks, embeddings, persist_directory="./chroma_db", collection_name="documents"):
    """Create a ChromaDB vector store from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"Created vector store with {len(chunks)} chunks at {persist_directory}")
    return vectorstore


def load_vectorstore(embeddings, persist_directory="./chroma_db", collection_name="documents"):
    """Load an existing ChromaDB vector store from disk."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


# --- Execution ---

if __name__ == "__main__":
    # Load cached docs (run load_claude_docs() + save_docs() first if not cached)
    docs = load_saved_docs("claude_docs/raw")

    # Clean and chunk
    docs = clean_docs(docs)
    chunks = chunk_documents(docs)

    # Embed and store in ChromaDB
    embeddings = get_ollama_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
