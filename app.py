import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import time
from dotenv import load_dotenv
load_dotenv()

## Debug: Print environment variables
print("Environment variables:", os.environ)

## Load the groq api key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    ## Debug: Print loaded documents
    print("Loaded documents:", st.session_state.docs)

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    ## Debug: Print split documents
    print("Split documents:", st.session_state.final_documents)

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
    ## Debug: Print vectors
    print("Vectors:", st.session_state.vectors)

st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key,
               model="Gemma-7b-It")

prompt_template = ChatPromptTemplate.from_template(
    """
Answer the Questions based on the provided context only. If you don't know the answer, just say that you don't know. Do not make up an answer.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Enter your question here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    
    ## Debug: Print response
    print("Response:", response)
    print("Response Time: ", time.process_time() - start)

    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------")
