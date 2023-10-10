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

file_path = "faiss_store_openai.pkl"

# Setting Up Streamlit Interface
st.title("News Research Tool")

## Streamlit Sidebar
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:
    # Loading Data from User-Inputted URLs
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading...Started...")     # Updating Progress Bar
    data = loader.load()

    # Spltting Data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter...Started...")     # Updating Progress Bar
    docs = text_splitter.split_documents(data)

    # Create Embeddings and Save it to FAISS Index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")     # Updating Progress Bar
    time.sleep(2)

    # Save the FAISS Index to a Pickle File
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Adding User Question Section
llm = OpenAI(temperature = 0.9, max_tokens = 500)       # Creating OpenAI LLM
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Reading Pickle File
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore.as_retriever())
            result = chain({'question': query}, return_only_outputs = True)

            ## Creating Answer Section
            st.header("Answer")
            st.write(result["answer"])

            # Display Sources if Available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")      # Split Sources by New Line
                for source in sources_list:
                    st.write(source)