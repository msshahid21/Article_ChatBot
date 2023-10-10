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

main_placefolder = st.empty()

if process_url_clicked:
    # Loading Data from User-Inputted URLs
    loader = UnstructuredURLLoader(urls = urls)
    main_placefolder.text("Data Loading...Started...")     # Updating Progress Bar
    data = loader.load()

    # Spltting Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placefolder.text("Text Splitter...Started...")     # Updating Progress Bar
    docs = text_splitter.split_documents(data)

    # Create Embeddings and Save it to FAISS Index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Started Building...")     # Updating Progress Bar
    time.sleep(2)

    # Save the FAISS Index to a Pickle File
    file_path = "faiss_store_openai.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)