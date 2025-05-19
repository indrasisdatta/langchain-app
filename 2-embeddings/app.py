from json import load
import os 
from dotenv import load_dotenv
from langchain_community.llms import Ollama 

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

# Langsmith tracking 
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACKING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# Prompt template 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant. Please respond to the questions asked."),
    ("user", "Questions: {question}")
])

# Streamlit app 
st.title("Langchain Demo with Llama 2")
input_text = st.text_input("What question do you have in mind?")

# Ollama Llama2 model 
llm = Ollama(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

             