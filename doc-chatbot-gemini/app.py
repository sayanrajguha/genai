#dependencies
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#load environment variables
load_dotenv()

#configure google gemini sdk
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#extract text from pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_stream = BytesIO(pdf)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extractText()
    return text

#chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#initialize vector store and save in local folder
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embeddings=embeddings)
    vector_store.save_local("faiss_index")

#define the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not present in the context just say, "Sorry, I am not aware of that. You can ask me something else.", dont provide the wrong answer\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:\n    
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",promt=prompt)
    return chain

#User Input
def user_input(user_question):
    #load embeddings from the model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #fetch locally stored vector db
    local_db = FAISS.load_local("faiss_index",embeddings=embeddings)
    #perform similarity search with user question
    docs = local_db.similarity_search(user_question)
    #load the conversational chain
    chain = get_conversational_chain()
    #feed the input documents and the user's question to the chain and obtain the output
    response = chain(
        {"input_documents" : docs, "question" : user_question},
        return_only_outputs=True
    )
    print(response)
    #write the output to streamlit ui
    st.write("\nReply:\t", response["output_text"])

#main function containing the streamlit ui app
def main():
    st.set_page_config("What's up, Doc?")
    st.page_title = "What's up, Doc? 🐰🥕"
    st.header("What's up, Doc? 🐰🥕")
    user_question = st.text_input("Ask a question from the PDF files you have uploaded")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files then click on Read", type=["pdf"], accept_multiple_files=True)
        if st.button("Read") and pdf_docs:
            if pdf_docs is not None:
                with st.spinner("Reading file..."):
                    bytesData = BytesIO(pdf_docs.getbuffer())
                    raw_text = get_pdf_text(bytesData)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done!")


if __name__ == "__main__":
    main()
