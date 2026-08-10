"""
Persistent FAISS index management for the AWS Lambda RAG backend.

Why this module exists
----------------------
Building a FAISS index requires downloading the PDF from S3, parsing it,
splitting it into chunks, and embedding every chunk with Bedrock. Doing that
on *every* Lambda invocation is slow and expensive.

Instead we split the work into two phases:

1. BUILD (run once): download PDF -> split -> embed -> build FAISS index ->
   serialize the index to S3. See `build_index()` and the `__main__` block.

2. QUERY (every invocation): load the serialized FAISS index from S3 once per
   warm container, then run similarity search + LLM generation. See
   `load_index()` and `query_index()`.

The FAISS index is stored in S3 as a single pickle file. Because the index is
read-only after build, it is safe to load it once and cache it in a module-level
global so warm Lambda invocations skip the S3 download entirely.
"""
import io
import os
import pickle
import tempfile

import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration (all overridable via Lambda environment variables)
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "dealerplatform")
PDF_KEY = os.environ.get("S3_OBJECT_KEY", "Leave-Policy-India.pdf")
INDEX_KEY = os.environ.get("S3_INDEX_KEY", "rag/faiss_index.pkl")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic.claude-v2")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Module-level cache so warm Lambda containers reuse the loaded index.
_INDEX_CACHE = None


# ---------------------------------------------------------------------------
# Embeddings / LLM helpers
# ---------------------------------------------------------------------------
def get_embeddings():
    """Return a BedrockEmbeddings client. Credentials come from the Lambda
    execution role (or the default AWS profile when run locally)."""
    return BedrockEmbeddings(model_id=EMBEDDING_MODEL)


def get_llm():
    """Return a Bedrock LLM client for answer generation."""
    return Bedrock(
        model_id=LLM_MODEL,
        model_kwargs={
            "max_tokens_to_sample": 300,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    )


# ---------------------------------------------------------------------------
# BUILD phase (run once)
# ---------------------------------------------------------------------------
def build_index(bucket=None, pdf_key=None, index_key=None):
    """Download the PDF from S3, split + embed it, build a FAISS index, and
    serialize it back to S3.

    Returns the FAISS vectorstore (also cached in memory).
    """
    bucket = bucket or S3_BUCKET
    pdf_key = pdf_key or PDF_KEY
    index_key = index_key or INDEX_KEY

    s3 = boto3.client("s3")

    # 1. Download the PDF to a local temp file (PyPDFLoader needs a local path).
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        s3.download_fileobj(bucket, pdf_key, tmp)
        tmp_path = tmp.name

    # 2. Load + split the document.
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed + build the FAISS index.
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    # 4. Serialize the index and upload it to S3.
    buffer = io.BytesIO()
    pickle.dump(vectorstore, buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, index_key)

    # Cache it so the same process can query immediately.
    global _INDEX_CACHE
    _INDEX_CACHE = vectorstore

    print(f"Index built and uploaded to s3://{bucket}/{index_key} "
          f"({len(chunks)} chunks).")
    return vectorstore


# ---------------------------------------------------------------------------
# QUERY phase (every invocation)
# ---------------------------------------------------------------------------
def load_index(bucket=None, index_key=None, force_reload=False):
    """Load the FAISS index from S3, caching it for warm invocations.

    Returns the FAISS vectorstore.
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is not None and not force_reload:
        return _INDEX_CACHE

    bucket = bucket or S3_BUCKET
    index_key = index_key or INDEX_KEY

    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, index_key, buffer)
    buffer.seek(0)

    _INDEX_CACHE = pickle.load(buffer)
    print(f"Index loaded from s3://{bucket}/{index_key}.")
    return _INDEX_CACHE


def query_index(question, vectorstore=None, k=4):
    """Run a similarity search against the vectorstore and generate an answer
    grounded in the retrieved chunks using the Bedrock LLM.

    Returns a dict with the answer and the retrieved source chunks.
    """
    vectorstore = vectorstore or load_index()

    # 1. Similarity search: find the k most relevant chunks.
    docs = vectorstore.similarity_search(question, k=k)

    # 2. Build a grounded prompt from the retrieved chunks.
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context provided below. Match the tone and style of the context.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    # 3. Generate the answer with the Bedrock LLM.
    llm = get_llm()
    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs],
    }


# ---------------------------------------------------------------------------
# CLI entry point for the BUILD phase
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run locally (or in a one-off Lambda) to build + persist the index:
    #   python rag_index.py
    build_index()
