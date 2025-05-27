import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

import os 

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")
# embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit 
st.title("Conversational Q&A Chatbot")
st.write("Upload PDF and chat with their content")

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="Gemma2-b-It")

session_id = "session_1"

if "store" not in st.session_state:
    st.session_state.store = {}

# File upload section
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the document 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_store")
    retriever = vector_store.as_retriever()

    contexualize_q_system_prompt = (
        """
        Given a chat history and the latest user question, 
        which might reference question in the chat history,
        formulate a standalone question which can be understood without the chat history.
        DO NOT answer the question. Just reformulate it if needed and otherwise return it as it is."""
    )

    contexualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contexualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contexualize_q_prompt)

    system_prompt = (
        """
        You are a helpful assistant for question-answering tasks.
        Use the following context to answer the question.
        If you don't know the answer, just say "I don't know".
        Use 3 sentences maximum and keep the answer short.
        \n\n
        {context} 
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", output_messages_key="chat_history")

    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Generating answer..."):
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": { "session_id": session_id },
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

