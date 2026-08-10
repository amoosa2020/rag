# AWS Lambda entry point for the RAG backend.
# Lambda's default handler is "lambda_function.lambda_handler", so this file
# MUST be named lambda_function.py and expose a function named lambda_handler.
import json

import rag_index as rag


def lambda_handler(event, context):
    """
    AWS Lambda handler.

    Expected event (API Gateway proxy integration):
        {
            "body": "{\"question\": \"How many casual leaves do I get?\"}"
        }

    Or a plain JSON event:
        {
            "question": "How many casual leaves do I get?"
        }

    The FAISS index is loaded once from S3 and cached for warm invocations,
    so each call only does a similarity search + LLM generation.
    """
    try:
        # Extract the question from the incoming event.
        question = _extract_question(event)

        if not question:
            return _response(400, {"error": "Missing 'question' in the request body."})

        # Load the pre-built FAISS index from S3 (cached across warm calls).
        vectorstore = rag.load_index()

        # Run similarity search + LLM generation.
        result = rag.query_index(question=question, vectorstore=vectorstore)

        return _response(200, result)

    except Exception as exc:  # noqa: BLE001 - surface any error to the caller
        return _response(500, {"error": str(exc)})


def _extract_question(event):
    """Pull the 'question' field out of either an API Gateway or plain event."""
    body = event.get("body", event)

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None

    if isinstance(body, dict):
        return body.get("question")

    return None


def _response(status_code, payload):
    """Build an API Gateway-compatible HTTP response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload),
    }
