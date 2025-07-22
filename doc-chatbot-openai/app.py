import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = ''

# Upload PDF File
st.header("What's up, DOC?")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a pdf file and start asking questions", type=["pdf"])

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

# Break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=510,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
#st.write(chunks)

# Generating embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Creating vector store
    vector_store = FAISS.from_texts(chunks,embeddings)

# Get user's question
    user_question = st.text_input("Type your question here")

# Do similarity search
    if user_question is not None:
        match = vector_store.similarity_search(user_question)
        st.write(match)
# Output results
