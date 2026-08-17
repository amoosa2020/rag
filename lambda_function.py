# AWS Lambda entry point for the RAG statement rephraser.
# Lambda's default handler is "lambda_function.lambda_handler", so this file
# MUST be named lambda_function.py and expose a function named lambda_handler.
import json

import rag_index as rag


def lambda_handler(event, context):
    """
    AWS Lambda handler.

    Accepts a STATEMENT and rephrases it following the style/format of the
    Job Aids document (2025 Dealer Profile Job Aid).

    Expected event (API Gateway proxy integration):
        {
            "body": "{\"statement\": \"The dealer profile needs updating.\"}"
        }

    Or a plain JSON event:
        {
            "statement": "The dealer profile needs updating."
        }

    The FAISS index (built from the Job Aids document) is loaded once from S3
    and cached for warm invocations, so each call only does a similarity search
    + LLM rephrasing.
    """
    try:
        # Extract the statement from the incoming event.
        statement = _extract_statement(event)

        if not statement:
            return _response(400, {"error": "Missing 'statement' in the request body."})

        # Load the pre-built FAISS index from S3 (cached across warm calls).
        vectorstore = rag.load_index()

        # Run similarity search + LLM rephrasing.
        result = rag.query_index(statement=statement, vectorstore=vectorstore)

        return _response(200, result)

    except Exception as exc:  # noqa: BLE001 - surface any error to the caller
        return _response(500, {"error": str(exc)})


def _extract_statement(event):
    """Pull the 'statement' field out of either an API Gateway or plain event."""
    body = event.get("body", event)

    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None

    if isinstance(body, dict):
        return body.get("statement")

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
