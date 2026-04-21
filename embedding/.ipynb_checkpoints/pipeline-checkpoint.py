import os
import nest_asyncio
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

nest_asyncio.apply()
os.environ.setdefault("USER_AGENT", "vector-db-rag-crawler/1.0")


# --- Document Loading ---

def load_from_sitemap(web_path, filter_urls=None):
    """Load documents from a sitemap URL, optionally filtering by URL patterns."""
    loader = SitemapLoader(
        web_path=web_path,
        filter_urls=filter_urls,
        parsing_function=lambda content: content.get_text(separator="\n", strip=True),
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


def get_huggingface_embeddings(model="mixedbread-ai/mxbai-embed-large-v1", device="cuda"):
    """Get embeddings via HuggingFace (local, free, GPU-accelerated)."""
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


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
