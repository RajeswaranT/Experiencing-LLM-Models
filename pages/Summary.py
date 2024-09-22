import streamlit as st

st.info("Gen AI project for experiencing multiple LLM models.")

with st.expander("Introduction", expanded=True):
    st.markdown(
        """
## Introduction

**TryLLM** This project to Generate context based multiple LLM models.
"""
    )

with st.expander("Tools Used"):
    st.markdown("""
## Tools Used

The following non-trivial tools were used in this application:

- `Python` (^3.10)
- `Streamlit`: For the UI
- `Assembly AI`: For Audio transcription
- `LangChain`: For generating critical analyses and recommendations using `OpenAI`'s LLM
- `Hugging Face`: For generating cover image ideas
""")
    
with st.expander("Features"):
    st.markdown("""
## Features

The application provides the following features:

**Experience LLM model** : It uses the LLM model to generate the content based on the user input.
- **RAG Q&A**: It uses the RAG model to answer questions Wikipedia, Arxiv and selected LLM model.
- **PDF Q&A**: Answer questions based on PDF content.
- **SQL DB Q&A**: Answer questions based on SQL database content.
- **Text Summarization**: Summarize the text content.
- **Youtube Vides Summarizations**: Summarize the video content.
- **Chatbot with Conversationa history**: Logic for incorporating historical messages and chat history.
""")
   