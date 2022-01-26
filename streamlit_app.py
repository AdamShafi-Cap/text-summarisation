# Imports
import pandas as pd
import numpy as np
import pickle

# Streamlit
import streamlit as st
import preshed
import cymem

# PDF
import sys
import io
import os

# Summarization using extractive bert
from summarizer import Summarizer, sentence_handler

# BERT based models for document search
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### App Set Up ###

st.set_page_config(layout="wide")
file, text, q, embeddings_option, num_pages = None, None, None, None, None

st.title('Search Guidance')
st.header('PIP Assessment Guide Part 2: The Asessment Criteria ')
st.write('''
This tool leverages advanced NLP techniques to search and summarise guidance documents. Enter your query below to see the most relevant sections of the document. Longer sections will be summarised. ''')

@st.cache(allow_output_mutation=True)
def load_qa_model():   
    try:
        qa = pickle.load(open('./dbert.pkl', 'rb'))
    except:
        qa = SentenceTransformer('sentence-transformers/multi-qa-distilbert-dot-v1')
        pickle.dump(qa, open('./dbert.pkl', 'wb'))
    return qa

@st.cache(allow_output_mutation=True, hash_funcs={preshed.maps.PreshMap:id, cymem.cymem.Pool:id})
def load_summariser_model():   
    summ = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
    return summ

@st.cache()
def load_data():
    paragraphs = pd.read_csv('paragraphs_clean.csv')
    paragraphs_embedded = pd.read_csv('paragraphs_embedded.csv')
    return paragraphs, paragraphs_embedded

def ask(q:str, X:pd.DataFrame, s:pd.DataFrame, n: int, model)->pd.Series:
    
    embedding = np.array(model.encode([q])[0])
        
    sorted_index = (X
                    .apply(lambda row: np.dot(row, embedding), axis=1)
                    .abs()
                    .sort_values(ascending=False)
                   )
    
    return s.loc[sorted_index.index].head(n)

def summarize(text, summarizer_model, n=1):
    result = summarizer_model(text, num_sentences=n,min_length=0)
    return result

def bold_sentences(text,summary):
    handler =  sentence_handler.SentenceHandler()
    bold = " ".join([f"__**{sentence}**__" 
                    if summary.find(sentence) != -1
                    else sentence 
                    for sentence in handler.process(text,min_length = 0)])
    return bold



paragraphs, paragraphs_embedded = load_data()

qa = load_qa_model()
summ = load_summariser_model()

q = st.text_input('What is your query?')
if q:
    ans = ask(q, X=paragraphs_embedded, s=paragraphs, n=3, model=qa)
    for i,t in ans.values:
        with st.beta_expander(f'Section: {i}'):
            if len(t)>60:
                summary = summarize(t, summ, 1)
                st.success(summary)
                st.write(bold_sentences(t,summary))
            else:
                st.write(t)