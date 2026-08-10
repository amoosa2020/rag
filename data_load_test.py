#1. Import Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
from langchain_community.document_loaders import PyPDFLoader
import requests

#2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
# PyPDFLoader only accepts a LOCAL file path, not a URL.
# So we first download the PDF to a local file, then load it.
url = 'https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf'
response = requests.get(url)
response.raise_for_status()  # raises an error if the download fails

with open('Leave-Policy-India.pdf', 'wb') as f:
    f.write(response.content)

data_load = PyPDFLoader('Leave-Policy-India.pdf')
data_test = data_load.load_and_split()
print(len(data_test))
print(data_test[0])
