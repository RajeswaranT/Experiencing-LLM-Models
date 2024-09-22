import streamlit as st
from constants import *
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

def main():
    supported_llm_models=""
    st.set_page_config(page_title="Experiencing LLM models.", page_icon="🦜")
    ## Sidebar for settings
    with st.sidebar:
        st.sidebar.title("Settings")
        selected_option=st.sidebar.radio(label="Choose the API Key",options=SUPPORTED_API_KEYS)
        api_key=""

        if SUPPORTED_API_KEYS.index(selected_option)==0:
            api_key=st.sidebar.text_input("Enter your Huggingface API Key:",type="password")
        elif SUPPORTED_API_KEYS.index(selected_option)==1:
            api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")
        elif SUPPORTED_API_KEYS.index(selected_option)==2:
            api_key=st.sidebar.text_input("Enter your OpenAI API Key:",type="password")
        elif SUPPORTED_API_KEYS.index(selected_option)==3:
            api_key=st.sidebar.text_input("Enter your AssemblyAI API Key:",type="password")
        st.divider()
        supported_llm_model=st.selectbox(
            "Choose the LLM foundational model",tuple(map(lambda x:x["name"],SUPPORTED_LLM_MODELS))
        )

        selected_feature=st.radio("Select the feature ",options=SUPPOTED_FEATURES)

    st.title(f"🔎 Experiencing '{supported_llm_model}' LLM models to Generate the {selected_feature}.") 
    
    if SUPPOTED_FEATURES.index(selected_feature)==0:
        rag_qa(supported_llm_model,api_key)
    elif SUPPOTED_FEATURES.index(selected_feature)==1:
        pdf_qa()
    elif SUPPOTED_FEATURES.index(selected_feature)==2:
        sql_db_qa()
    elif SUPPOTED_FEATURES.index(selected_feature)==3:
        text_summarization()
    elif SUPPOTED_FEATURES.index(selected_feature)==4:
        youtube_videos_summarizations()
    elif SUPPOTED_FEATURES.index(selected_feature)==5:
        chatbot_with_conversationa_history()
    
def rag_qa(supported_llm_model,api_key):
    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg['content'])
    
    if prompt:=st.chat_input(placeholder="What is machine learning?"):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        llm=ChatGroq(groq_api_key=api_key,model_name=supported_llm_model,streaming=True)
        tools=[search,arxiv,wiki]

        search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write(response)

def pdf_qa():
    st.write("PDF Q&A")

def sql_db_qa():
    st.write("SQL DB Q&A")

def text_summarization():
    st.write("Text Summarization")

def youtube_videos_summarizations():
    st.write("Youtube Vides Summarizations")    

def chatbot_with_conversationa_history():
    st.write("Chatbot with Conversationa history")  


if __name__ == "__main__":
    load_dotenv()
    main()