import streamlit as st  
from pathlib import Path 
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase 
from langchain.agents.agent_types import AgentType 
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine 

import os
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Langchain - Chat with SQL DB")
st.title("Langchain - Chat with SQL DB")

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=api_key, model="llama3-8b-8192", streaming=True)

def configure_db():
    print("DB connection string:") 
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
    return SQLDatabase(create_engine(conn_str))

db = configure_db()
print(db)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm, 
    toolkit=toolkit,
    verboze=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
# st.sidebar.button("Clear message history")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ] 

for msg in st.session_state.messages: 
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})

