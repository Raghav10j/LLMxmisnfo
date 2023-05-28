from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader

from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd


# loader = UnstructuredFileLoader('content.txt')
# raw_documents = loader.load()

with open('content.txt', 'r') as file:
    data = file.read()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
# docs = ''.join(raw_documents)
# metadatas = []
texts = text_splitter.split_text(data)
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
vectorstore.save_local("vectorstore_covid")

