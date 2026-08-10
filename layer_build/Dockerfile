# Build a Lambda layer for the RAG backend dependencies.
# Uses the official AWS Lambda Python 3.13 base image so the compiled
# binaries (faiss-cpu, etc.) are compatible with the Lambda runtime.
FROM public.ecr.aws/lambda/python:3.13

# Install the dependencies into /opt/python (the Lambda layer path).
RUN pip install \
    langchain-classic==1.0.8 \
    langchain-community==0.4.2 \
    langchain-core==1.5.3 \
    langchain-text-splitters==1.1.2 \
    pypdf>=4.0.0 \
    faiss-cpu>=1.8.0 \
    boto3>=1.34.0 \
    --target /opt/python

# Remove caches to keep the layer small (use python, not find, since the
# Lambda base image does not ship the `find` utility).
RUN python -c "import shutil, os; [shutil.rmtree(p) for p,_,_ in os.walk('/opt/python') if '__pycache__' in p]"
