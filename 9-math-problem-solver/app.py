import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain 
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Groq initialize
import os 
from dotenv import load_dotenv
load_dotenv()
# Disable SSL cert on local 
os.environ.pop("SSL_CERT_FILE", None)
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Streamlit app
st.set_page_config(page_title="Text to Math problem solver and data search assistant")
st.title("Text to Math problem solver using Google Gemma 2")

# Initialize the tools 
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet for various topics"
)

# Initialize the Math tool 
math_chain = LLMMathChain.from_llm(llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering Math related questions. Only input Mathematical expressions."
)

prompt = """
You're an agent tasked at solving mathematical questions. Logically arrive at the solution and explain the solution in pointwise format for the question below.
Question: {question}
"""

prompt_template = PromptTemplate(template=prompt, input_variables=["question"])

# Combine all tools into chain 
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logical questions"
)

# Initialize the agents 
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=False
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can provide solution to your Mathematical problems"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Function to generate response
def generate_response(user_question):
    response = assistant_agent.invoke({'input': user_question})
    return response 

# Start the interaction 
question = st.text_area("Enter your question", key="user_input")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({'role': 'user', 'content': question})
            st.chat_message('user').write(question)

            # As agent processes your request, this handler updates UI in real time
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # Execute Langchain agent, passing the chat history and use callback handler to show progress 
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role': 'assistant', 'content': response})

            st.write('Response: ')
            st.success(response)

            # Set flag to clear input on next run
            st.session_state.clear_input = True

            
