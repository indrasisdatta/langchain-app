import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context provided. 
    Please provide the most accurate response based on the question. 
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding(text):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loaders = PyPDFDirectoryLoader("research_papers") 
        st.session_state.docs = st.session_state.loaders.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and Llama3")

user_prompt = st.text_input("Enter your question from the research papers")

if st.button("Document Embedding"):
    with st.spinner("Creating vector embedding..."):
        create_vector_embedding(user_prompt)
        st.success("Vector database is ready!")

import time

if user_prompt:
    with st.spinner("Generating answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time() 
        response = retrieval_chain.invoke({"input": user_prompt})
        print(f"Response time: {time.process_time() - start} seconds")

        st.write(response["answer"])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("----------")