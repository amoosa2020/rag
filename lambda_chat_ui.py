"""
Streamlit chat UI that invokes the rag_backend AWS Lambda.

This UI sends the user's statement to the rag_backend Lambda (deployed in
us-east-2) via boto3 and displays the rephrased statement plus the RAG
source chunks used to generate it.

Run with:
    streamlit run lambda_chat_ui.py
"""
import json

import boto3
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration - matches the deployed Lambda
# ---------------------------------------------------------------------------
LAMBDA_FUNCTION_NAME = "rag_backend"
LAMBDA_REGION = "us-east-2"

st.set_page_config(
    page_title="RAG Rephraser Chat",
    page_icon="🤖",
    layout="wide",
)


def invoke_lambda(statement: str) -> dict:
    """Invoke the rag_backend Lambda with the given statement and return its body."""
    client = boto3.client("lambda", region_name=LAMBDA_REGION)

    payload = json.dumps({"statement": statement}).encode("utf-8")

    response = client.invoke(
        FunctionName=LAMBDA_FUNCTION_NAME,
        InvocationType="RequestResponse",
        Payload=payload,
    )

    # The Lambda returns an API Gateway-style envelope.
    envelope = json.loads(response["Payload"].read().decode("utf-8"))

    status_code = envelope.get("statusCode", 200)
    body = envelope.get("body", envelope)

    if isinstance(body, str):
        body = json.loads(body)

    if status_code >= 400:
        raise RuntimeError(f"Lambda returned status {status_code}: {body}")

    return body


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-family: sans-serif;
            color: #1f6feb;
            font-size: 42px;
            font-weight: 700;
        }
        .sub-title {
            font-family: sans-serif;
            color: #57606a;
            font-size: 16px;
        }
        .rephrase-box {
            background-color: #f0f6ff;
            border-left: 5px solid #1f6feb;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 12px;
        }
        .source-box {
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 13px;
            color: #24292f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🤖 RAG Rephraser Chat</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Invokes the <code>rag_backend</code> AWS Lambda '
    f"({LAMBDA_FUNCTION_NAME} @ {LAMBDA_REGION}) to rephrase statements using "
    "the Caterpillar Dealer Profile Job Aid.</p>",
    unsafe_allow_html=True,
)

# Initialize chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
            for src in msg.get("sources", []):
                with st.expander("📄 Source chunk"):
                    st.markdown(src)
        else:
            st.markdown(msg["content"])

# Chat input.
prompt = st.chat_input("Enter a statement to rephrase (e.g. 'The dealer profile needs updating.')")

if prompt:
    # Add user message to history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the Lambda.
    with st.chat_message("assistant"):
        with st.spinner("📢 Rephrasing via rag_backend Lambda..."):
            try:
                result = invoke_lambda(prompt)

                rephrased = result.get("rephrased_statement", "")
                sources = result.get("sources", [])

                st.markdown(
                    f'<div class="rephrase-box"><b>Rephrased statement:</b><br>'
                    f"{rephrased}</div>",
                    unsafe_allow_html=True,
                )

                if sources:
                    st.markdown("**Sources used:**")
                    for src in sources:
                        with st.expander("📄 Source chunk"):
                            st.markdown(src)

                # Persist assistant response in history.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rephrased,
                        "sources": sources,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface errors to the user
                st.error(f"❌ Failed to invoke Lambda: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {exc}"}
                )
