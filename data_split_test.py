#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
from langchain_community.document_loaders import PyPDFLoader   
from langchain_text_splitters import RecursiveCharacterTextSplitter

#2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
data_load=PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
#3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
data_sample = 'Welcome to the most comprehensive AWS Cloud Development Kit (CDK) - V2 on Udemy from an instructor with actual enterprise hands-on implementation experience migrating large number of workloads for Fortune 100 companies using AWS CDK V2.'
data_split_test = data_split.split_text(data_sample)
print(data_split_test)

#4. Create Embeddings -- Client connection
#5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
#5b  Create index for HR Report
#5c. Wrap within a function
#6a. Write a function to connect to Bedrock Foundation Model
#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html
