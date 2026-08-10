#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
import tempfile
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_community.llms import Bedrock
 
#5c. Wrap within a function
def hr_index():
    #2. Define the data source and load data with PDFLoader.
    # PyPDFLoader only accepts a LOCAL file path, not an S3 URI.
    # So we first download the PDF from S3 to a local temp file, then load it.
    # Configure the bucket and object key via environment variables so the
    # same code works in Lambda and locally.
    bucket = os.environ.get('S3_BUCKET', 'dealerplatform')
    object_key = os.environ.get('S3_OBJECT_KEY', 'Leave-Policy-India.pdf')

    s3 = boto3.client('s3')
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        s3.download_fileobj(bucket, object_key, tmp)
        tmp_path = tmp.name

    data_load = PyPDFLoader(tmp_path)
 
    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
    #4. Create Embeddings -- Client connection
    # In Lambda, credentials come from the execution role (no profile file exists).
    # Locally, boto3 falls back to the default profile automatically.
    data_embeddings=BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v1')
    #5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    #5b  Create index for HR Policy Document
    db_index=data_index.from_loaders([data_load])
    return db_index
#6a. Write a function to connect to Bedrock Foundation Model - Claude Foundation Model
def hr_llm():
    # In Lambda, credentials come from the execution role (no profile file exists).
    llm=Bedrock(
        model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":300,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm
#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def hr_rag_response(index,question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html