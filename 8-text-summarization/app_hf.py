import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq 
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

import os
from dotenv import load_dotenv 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_api_key = os.getenv("HUGGINGFACE_TOKEN")

# Streamlit App 
st.set_page_config(page_title="Langchain: Summarize Text from YouTube or Website")
st.title("Langchain: Summarize Text from YouTube or Website")
st.subheader("Summarize URL")

url = st.text_input("URL", label_visibility="collapsed")

# llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    max_length=150, 
    token=hf_api_key
)

prompt_template = """  
Provide summary for the following content in 300 words: 
Content: {text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the content from YouTube or Website"):
    # Validate all inputs 
    if not url.strip():
        st.error("Please enter a URL")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                else: 
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Chain for summarization 
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
