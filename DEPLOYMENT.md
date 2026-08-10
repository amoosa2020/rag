# AWS Lambda RAG Backend — Deployment Guide

This project implements a **Retrieval-Augmented Generation (RAG)** backend that runs
on AWS Lambda, uses **Amazon Bedrock** for embeddings + LLM, and **FAISS** as the
vector database. The FAISS index is persisted to **S3** so it is built once and
reused across invocations.

## Architecture

```
                    BUILD PHASE (run once)
   S3 (PDF) --download--> PyPDFLoader --> TextSplitter --> BedrockEmbeddings
                                                                  |
                                                                  v
   S3 (faiss_index.pkl) <--pickle-- FAISS.from_documents <--------+

                    QUERY PHASE (every invocation)
   API Gateway --> Lambda handler --> load FAISS index from S3 (cached)
                                          |
                                          v
   Question --> similarity_search --> top-k chunks --> Bedrock LLM --> Answer
```

## Files

| File | Purpose |
|------|---------|
| [`rag_index.py`](rag_index.py) | Builds/loads the FAISS index and runs the RAG query. |
| [`lambda_function.py`](lambda_function.py) | Lambda entry point (`lambda_handler`). |
| [`requirements.txt`](requirements.txt) | Python dependencies for the layer. |

## Prerequisites

1. An **S3 bucket** containing your PDF (e.g. `s3://dealerplatform/Leave-Policy-India.pdf`).
2. **Amazon Bedrock** access enabled in your region, with the models you use
   (`amazon.titan-embed-text-v1` and `anthropic.claude-v2`) **requested/approved**
   in the Bedrock console.
3. AWS CLI configured locally (for the build step).

## Step 1 — Build the index (run once)

Run the build script locally with your AWS credentials:

```bash
# Set env vars (optional; defaults are in rag_index.py)
export S3_BUCKET=dealerplatform
export S3_OBJECT_KEY=Leave-Policy-India.pdf
export S3_INDEX_KEY=rag/faiss_index.pkl

python rag_index.py
```

This downloads the PDF from S3, splits + embeds it, builds the FAISS index, and
uploads it to `s3://<bucket>/rag/faiss_index.pkl`.

> **Alternative:** you can also trigger the build from inside Lambda by calling
> `rag.build_index()` once (e.g. via a test event), but running it locally is
> simpler and cheaper.

## Step 2 — Create the Lambda layer (dependencies)

FAISS and LangChain are large, so put them in a **Lambda layer** rather than
inlining them in the function code.

```bash
# On a machine with the same OS/arch as Lambda (Amazon Linux 2 / x86_64),
# or use a Docker image with the public.ecr.aws/lambda/python:3.11 base.
mkdir -p layer/python
pip install -r requirements.txt -t layer/python
# Remove unnecessary files to keep the layer small
find layer/python -name "*.pyc" -delete
find layer/python -type d -name "__pycache__" -exec rm -rf {} +

# Zip the layer
cd layer && zip -r ../rag-layer.zip python && cd ..
```

Upload `rag-layer.zip` as a **Lambda layer** (Python 3.11, x86_64).

## Step 3 — Create the Lambda function

1. Create a new Lambda function (runtime **Python 3.11**, architecture **x86_64**).
2. Upload `lambda_function.py` and `rag_index.py` as the function code
   (zip them together, or use the console editor to paste both files).
3. Attach the **layer** created in Step 2.
4. Set the handler to `lambda_function.lambda_handler`.
5. Set the **environment variables**:

   | Variable | Example | Description |
   |----------|---------|-------------|
   | `S3_BUCKET` | `dealerplatform` | Bucket holding the PDF and the index. |
   | `S3_OBJECT_KEY` | `Leave-Policy-India.pdf` | PDF object key (build only). |
   | `S3_INDEX_KEY` | `rag/faiss_index.pkl` | Where the FAISS index is stored. |
   | `EMBEDDING_MODEL` | `amazon.titan-embed-text-v1` | Bedrock embedding model. |
   | `LLM_MODEL` | `anthropic.claude-v2` | Bedrock generative model. |
   | `CHUNK_SIZE` | `1000` | Text splitter chunk size (build only). |
   | `CHUNK_OVERLAP` | `200` | Text splitter overlap (build only). |

6. Increase the **timeout** to at least **60 seconds** (the first cold load of
   the FAISS index can take a few seconds) and set **memory** to **1024 MB** or
   more (FAISS + LangChain are memory-hungry).

## Step 4 — IAM permissions

Attach an execution role to the Lambda with at least:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::dealerplatform/*"
    },
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "*"
    }
  ]
}
```

## Step 5 — Test

Invoke the function with a plain JSON event:

```json
{
  "question": "How many casual leaves do I get?"
}
```

Or via API Gateway (proxy integration):

```json
{
  "body": "{\"question\": \"How many casual leaves do I get?\"}"
}
```

Expected response:

```json
{
  "statusCode": 200,
  "body": "{\"answer\": \"...\", \"sources\": [\"...\"]}"
}
```

## Notes

- **Warm vs cold starts:** The FAISS index is loaded from S3 on the first
  invocation of a container and cached in a module-level global, so subsequent
  warm invocations skip the S3 download.
- **Rebuilding the index:** If the PDF changes, re-run Step 1 to overwrite the
  index in S3. No code changes needed.
- **Vector DB choice:** FAISS is used here because it is lightweight and can be
  fully serialized to S3. For very large corpora or multi-user concurrency,
  consider Amazon OpenSearch Serverless or Pinecone instead.
