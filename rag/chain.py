from langchain_core.prompts import ChatPromptTemplate
from embedding.pipeline import get_ollama_embeddings, load_vectorstore


# --- Retriever ---

def get_retriever(persist_directory="./chroma_db", collection_name="documents", k=4):
    """Create a retriever from an existing ChromaDB vector store.

    The retriever takes a query string and returns the top-k most relevant
    document chunks based on embedding similarity.
    """
    embeddings = get_ollama_embeddings()
    vectorstore = load_vectorstore(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# --- Prompt Template ---

RAG_PROMPT = """You are a helpful assistant answering questions about Claude Code.
Use the following retrieved context to answer the question.
If the context doesn't contain the answer, say you don't know — don't make anything up.

Context:
{context}

Question: {question}

Answer:"""


def get_prompt():
    """Create a chat prompt template for the RAG chain.

    The template has two input variables:
    - {context}: retrieved document chunks, joined as a string
    - {question}: the user's query
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT)
