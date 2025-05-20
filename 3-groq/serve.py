from ast import Str
from fastapi import FastAPI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Create prompt template 
system_template = "Translate the following text to {language}."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])
parser = StrOutputParser()

chain = prompt_template | model | parser 

# App definition

app = FastAPI(
    title="Langchain Groq API", 
    description="Langchain Groq API", 
    version="0.0.1"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000) 