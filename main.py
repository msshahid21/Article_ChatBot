import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Importing OpenAPI Enviroment Key
from secret_key import api_key
os.environ['OPENAI_API_KEY'] = api_key

# Setting Up Streamlit Interface
st.title("News Research Tool")

## Streamlit Sidebar
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # Loading Data from User-Inputted URLs
    loader = UnstructuredURLLoader(urls = urls)
    data = loader.load()

    # Spltting Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    docs = text_splitter.split_documents(data)

    # Create Embeddings and Save it to FAISS Index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Save the FAISS Index to a Pickle File
    file_path = "faiss_store_openai.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)