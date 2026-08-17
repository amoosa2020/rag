# AWS Lambda RAG Statement Rephraser — Deployment Guide

This project implements a **Retrieval-Augmented Generation (RAG)** application that runs
on AWS Lambda, uses **Amazon Bedrock** for embeddings + LLM, and **FAISS** as the
vector database. The FAISS index is built **once** from the **2025 Dealer Profile Job Aid**
document (a PowerPoint) and persisted to **S3**, then reused across invocations.

The application **rephrases an incoming statement** following the style/format suggested
in the Job Aids document. It uses **only the Job Aids embeddings** (not the Leave Policy).

## Architecture

```
                    BUILD PHASE (run on a schedule)
   S3 (Job Aid pptx/pdf) --download--> Loader (PDF or PPTX) --> TextSplitter
                                                                     |
                                                                     v
   S3 (faiss_index.pkl) <--pickle-- FAISS.from_documents <-- BedrockEmbeddings
        ^
        |  (build_embedding.py, triggered by EventBridge schedule)
        |
                    QUERY PHASE (every invocation)
   API Gateway --> Lambda handler --> load FAISS index from S3 (cached)
                                           |
                                           v
   Statement --> similarity_search --> top-k Job Aids chunks --> Bedrock LLM
                                                                     |
                                                                     v
                                                          Rephrased statement
```

## Files

| File | Purpose |
|------|---------|
| [`build_embedding.py`](build_embedding.py) | **Standalone build script.** Downloads the source document (PDF or PPTX) from S3, splits + embeds it, builds the FAISS index, and uploads it to S3. Runnable on a schedule (EventBridge) or manually. |
| [`rag_index.py`](rag_index.py) | Loads the FAISS index and runs the RAG rephrasing query (`load_index`, `query_index`). Also contains `load_document` (PDF/PPTX loader) and `build_index`. |
| [`lambda_function.py`](lambda_function.py) | Lambda entry point (`lambda_handler`). Accepts a `statement` and returns the rephrased statement. |
| [`requirements.txt`](requirements.txt) | Python dependencies for the layer (includes PPTX support). |
| [`layer_build/Dockerfile`](layer_build/Dockerfile) | Builds the Lambda layer with all dependencies. |

## Prerequisites

1. An **S3 bucket** (e.g. `dealerplatform`) containing the source document, e.g.
   `s3://dealerplatform/2025 Dealer Profile Job Aid.pptx`. Both **PDF** and **PPTX**
   are supported.
2. **Amazon Bedrock** access enabled in your region, with the models you use
   (`amazon.titan-embed-text-v1`/`v2` for embeddings and an **Amazon Nova**
   model such as `amazon.nova-lite-v1:0` for generation) **requested/approved**
   in the Bedrock console.
3. AWS CLI configured locally (for the manual build step).

## Step 1 — Build the index (run once or on a schedule)

The index is built by the standalone [`build_embedding.py`](build_embedding.py) script.
It is **not** built on every Lambda invocation.

### Option A — Run manually (local)

```bash
# Set env vars (optional; defaults are in build_embedding.py)
export S3_BUCKET=dealerplatform
export S3_OBJECT_KEY="2025 Dealer Profile Job Aid.pptx"
export S3_INDEX_KEY=rag/faiss_index.pkl

python build_embedding.py
```

This downloads the source document from S3, splits + embeds it, builds the FAISS index,
and uploads it to `s3://<bucket>/rag/faiss_index.pkl`.

### Option B — Run on a schedule (EventBridge, recommended)

Create a **build Lambda** that wraps `build_embedding.py` and trigger it with an
**EventBridge schedule rule** so the index is rebuilt automatically whenever the source
document changes.

1. Create a Lambda function (runtime **Python 3.13**, architecture **x86_64**) with
   `build_embedding.py` as the code and the **layer** from Step 2 attached.
   > **Important:** The layer is built with the Python 3.13 Lambda base image
   > (`public.ecr.aws/lambda/python:3.13`), so the function runtime MUST be
   > **Python 3.13**. Using 3.11 will fail to load the layer's compiled
   > `.cpython-313` extensions.
2. Set the handler to `build_embedding.lambda_handler`.
3. Set the same environment variables as above.
4. Create an **EventBridge rule** (e.g. daily or weekly) with a cron expression:

   ```text
   cron(0 2 * * ? *)   # every day at 02:00 UTC
   ```

   Target the build Lambda. Grant EventBridge permission to invoke it.

> **Note:** The build Lambda needs `s3:GetObject` (download source) and `s3:PutObject`
> (upload index) plus `bedrock:InvokeModel` — see Step 4.

## Step 2 — Create the Lambda layer (dependencies)

FAISS, LangChain, and the PPTX parser are large, so put them in a **Lambda layer**
rather than inlining them in the function code.

```bash
# On a machine with the same OS/arch as Lambda (Amazon Linux 2 / x86_64),
# or use the Docker image in layer_build/Dockerfile.
mkdir -p layer/python
pip install -r requirements.txt -t layer/python
# Remove unnecessary files to keep the layer small
find layer/python -name "*.pyc" -delete
find layer/python -type d -name "__pycache__" -exec rm -rf {} +

# Zip the layer
cd layer && zip -r ../rag-layer.zip python && cd ..
```

Upload `rag-layer.zip` as a **Lambda layer** (Python 3.13, x86_64).

## Step 3 — Create the query Lambda function

1. Create a new Lambda function (runtime **Python 3.13**, architecture **x86_64**).
   > **Important:** Must be **Python 3.13** to match the layer's compiled
   > `.cpython-313` extensions.
2. Upload `lambda_function.py` and `rag_index.py` as the function code
   (zip them together, or use the console editor to paste both files).
3. Attach the **layer** created in Step 2.
4. Set the handler to `lambda_function.lambda_handler`.
5. Set the **environment variables**:

   | Variable | Example | Description |
   |----------|---------|-------------|
   | `S3_BUCKET` | `dealerplatform` | Bucket holding the source doc and the index. |
   | `S3_OBJECT_KEY` | `2025 Dealer Profile Job Aid.pptx` | Source document object key (build only). |
   | `S3_INDEX_KEY` | `rag/faiss_index.pkl` | Where the FAISS index is stored. |
   | `EMBEDDING_MODEL` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model. |
   | `LLM_MODEL` | `amazon.nova-lite-v1:0` | Bedrock generative model. **Must be an Amazon Nova model** (`amazon.nova-*`) — the code calls the Nova Messages API. A Claude model id will fail. |
   | `CHUNK_SIZE` | `1000` | Text splitter chunk size (build only). |
   | `CHUNK_OVERLAP` | `200` | Text splitter overlap (build only). |

6. Increase the **timeout** to at least **60 seconds** (the first cold load of
   the FAISS index can take a few seconds) and set **memory** to **1024 MB** or
   more (FAISS + LangChain are memory-hungry).

## Step 4 — IAM permissions

Attach an execution role to the **query** Lambda with at least:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
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

The **build** Lambda additionally needs `s3:PutObject` (to upload the index):

```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject"],
  "Resource": "arn:aws:s3:::dealerplatform/*"
}
```

## Step 5 — Test

Invoke the query function with a plain JSON event:

```json
{
  "statement": "The dealer profile needs to be updated with the new contact details."
}
```

Or via API Gateway (proxy integration):

```json
{
  "body": "{\"statement\": \"The dealer profile needs to be updated with the new contact details.\"}"
}
```

Expected response:

```json
{
  "statusCode": 200,
  "body": "{\"rephrased_statement\": \"...\", \"sources\": [\"...\"]}"
}
```

## Notes

- **Warm vs cold starts:** The FAISS index is loaded from S3 on the first
  invocation of a container and cached in a module-level global, so subsequent
  warm invocations skip the S3 download.
- **Rebuilding the index:** When the Job Aids document changes, re-run
  `build_embedding.py` (or let the EventBridge schedule do it) to overwrite the
  index in S3. No code changes needed.
- **Source document types:** Both **PDF** (`.pdf`) and **PowerPoint** (`.pptx`/`.ppt`)
  are supported. The loader is chosen automatically by file extension.
- **Vector DB choice:** FAISS is used here because it is lightweight and can be
  fully serialized to S3. For very large corpora or multi-user concurrency,
  consider Amazon OpenSearch Serverless or Pinecone instead.
