# Imports
import pandas as pd
import numpy as np
import pickle
from stqdm import stqdm

# Streamlit
import streamlit as st
import preshed
import cymem

# PDF
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

# Summarization using extractive bert
from summarizer import Summarizer, sentence_handler
#import tensorflow_hub as hub

# BERT based models for document search
from sentence_transformers import SentenceTransformer


### App Set Up ###

stqdm.pandas()
st.set_page_config(layout="wide")
file, text, q, embeddings_option, num_pages = None, None, None, None, None

st.title('Project Pico: Natural Language Processing (NLP) Demo')
st.header('Applications of BERT Models')
st.write('''
BERT is an open-source model for processing natural language, developed by Google. It was designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.''')
st.write('''
This app uses BERT models trained for specific tasks to demonstrate the model's ability to a) find relevant passages in a document, given a query and b) find representative sentences within the passage to create a summary. To use it, either upload a document or use the example Brexit document. You must then select a model from the list of BERT variants and enter a query. The app will use the model you select to search the query against the document. It then uses a separate BERT model to write a summary using sentences from the paragraph.''' )


