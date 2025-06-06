import streamlit as st  
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType 
from langchain.callbacks import StreamlitCallbackHandler

import os 
from dotenv import load_dotenv
load_dotenv(override=True)   
api_key = os.getenv("GROQ_API_KEY")
print(api_key)

# Disable SSL cert on local 
os.environ.pop("SSL_CERT_FILE", None)

# Arxiv and Wikipedia tools 
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(arxiv_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)  

search = DuckDuckGoSearchRun(name="Search")

st.title("Langchain - Chat with Search")

if "messages" not in st.session_state: 
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Conversation history
for msg in st.session_state.messages: 
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
if prompt:= st.chat_input(placeholder="What is Machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Load Groq LLM
    llm = ChatGroq(groq_api_key=api_key, model="llama3-8b-8192" )
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)

    # Render assistant's response in Streamlit Chat UI block
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)




