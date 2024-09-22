import validators,streamlit as st
from constants import *
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")

def main():
    
    supported_llm_model=""
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
    
    if not api_key.strip():
        st.error("Please provide the API Key to get started")
        return

    if SUPPOTED_FEATURES.index(selected_feature)==0:
        rag_qa(supported_llm_model,api_key)
    elif SUPPOTED_FEATURES.index(selected_feature)==1:
        pdf_qa()
    elif SUPPOTED_FEATURES.index(selected_feature)==2:
        sql_db_qa()
    elif SUPPOTED_FEATURES.index(selected_feature)==3:
        text_summarization()
    elif SUPPOTED_FEATURES.index(selected_feature)==4:
        youtube_videos_summarizations(supported_llm_model,api_key)
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

def chatbot_with_conversationa_history():
    st.write("Youtube Vides Summarizations")    

def youtube_videos_summarizations(supported_llm_model,api_key):
    st.subheader('Summarize URL')
    generic_url=st.text_input("URL",label_visibility="collapsed")
    ## Gemma Model USsing Groq API
    llm =ChatGroq(model=supported_llm_model, groq_api_key=api_key)

    prompt_template="""
    Provide a summary of the following content in 300 words:
    Content:{text}

    """
    prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        ## Validate all the inputs
        if not api_key.strip() or not generic_url.strip():
            st.error("Please provide the information to get started")
        elif not validators.url(generic_url):
            st.error("Please enter a valid Url. It can may be a YT video utl or website url")
        else:
            try:
                with st.spinner("Waiting..."):
                    ## loading the website or yt video data
                    if "youtube.com" in generic_url:
                        loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                    else:
                        loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()

                    ## Chain For Summarization
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    output_summary=chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception:{e}")


if __name__ == "__main__":
    load_dotenv()
    main()