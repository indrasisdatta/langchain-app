from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os
from dotenv import load_dotenv  
load_dotenv() 

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_TRACING_V2"] = "true"

# Prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, engine, temperature, max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


engine = st.sidebar.selectbox("Select Model", ["llama3.2:1b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=300, value=150)

st.write("Ask your question")
user_input = st.text_input("")

if user_input:
    with st.spinner("Generating answer..."):
        response = generate_response(user_input, engine, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide the user input")



