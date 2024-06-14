#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai
# Text Splitting Utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import CrossEncoder

import streamlit as st
from pypdf import PdfReader
import numpy as np
import docx
import os

st.set_page_config(page_title="Document Query Assistant with Reranker and Completely Free", layout="wide")

def rank_doc(query, text_chunks, topN=5):
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = reranker.predict([[query, doc] for doc in text_chunks])
    top_indices = np.argsort(scores)[::-1][:topN]
    top_pairs = [text_chunks[index] for index in top_indices]
    return top_pairs

def rag(query, retrieved_documents, api_key):
    model = "gpt-4"

    openai.api_key = api_key

    information = "\n\n".join(retrieved_documents)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual 10K report."
                       "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message['content']
    return content

@st.cache_data
def process_pdf_texts(pdf_file):
    reader = PdfReader(pdf_file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    return clean_text_list(character_split_texts)

@st.cache_data
def process_docx_texts(docx_file):
    doc = docx.Document(docx_file)
    docx_texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(docx_texts))
    return clean_text_list(character_split_texts)

@st.cache_data
def process_txt_texts(txt_file):
    txt_texts = txt_file.read().decode("utf-8").splitlines()
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
    character_split_texts = character_splitter.split_text('\n\n'.join(txt_texts))
    return clean_text_list(character_split_texts)

def clean_text_list(text_list):
    cleaned_texts = []
    for text in text_list:
        text = text.replace('\t', ' ').replace('\n', ' ')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Choose a document file", type=['pdf', 'docx', 'txt'])

if uploaded_file and api_key:
    file_type = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_type == '.pdf':
        formatted_texts = process_pdf_texts(uploaded_file)
    elif file_type == '.docx':
        formatted_texts = process_docx_texts(uploaded_file)
    elif file_type == '.txt':
        formatted_texts = process_txt_texts(uploaded_file)
    st.session_state.processed_texts = formatted_texts

st.title("Free Document Query Assistant with Reranker")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.chat_history:
    for query, response in st.session_state.chat_history:
        st.container().markdown(f"**Q**: {query}")
        st.container().markdown(f"**A**: {response}")

query = st.text_input("Type your question here:", key="query")

if st.button("Submit Query"):
    if 'processed_texts' in st.session_state and query and api_key:
        with st.spinner('Processing...'):
            retrieved_documents = rank_doc(query, st.session_state.processed_texts)
            output_wrapped = rag(query, retrieved_documents, api_key)
            st.session_state.chat_history.append((query, output_wrapped))
            st.container().markdown(f"**Q**: {query}")
            st.container().markdown(f"**A**: {output_wrapped}")
    else:
        st.error("Please upload a document, ensure the API key is set, and type a question.")


# In[ ]:





# In[ ]:




