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

st.title('Title')
st.header('Header')
st.write('''
Text''')


sample_phrase = '''
The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
Real estate firm Tishman Speyer had owned the other 10%.
The buyer is RFR Holding, a New York real estate company.
Officials with Tishman and RFR did not immediately respond to a request for comments.
It's unclear when the deal will close.
The building sold fairly quickly after being publicly placed on the market only two months ago.
The sale was handled by CBRE Group.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
Meantime, rents in the building itself are not rising nearly that fast.
While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
Still the building is among the best known in the city, even to people who have never been to New York.
It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.
'''

@st.cache()
def load_models():   
    qa = SentenceTransformer('sentence-transformers/multi-qa-distilbert-dot-v1')
    summ = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
    return qa, summ

@st.cache()
def load_data():
    paragraphs = pd.read_csv('paragraphs.csv')
    paragraphs_embedded = pd.read_csv('paragraphs_embedded.csv')
    return paragraphs, paragraphs_embedded

@st.cache()
def ask(q:str, X:pd.DataFrame, s:pd.DataFrame, n: int, model)->pd.Series:
    
    embedding = np.array(model.encode([q])[0])
        
    sorted_index = (X
                    .apply(lambda row: np.dot(row, embedding), axis=1)
                    .abs()
                    .sort_values(ascending=False)
                   )
    
    return s.loc[sorted_index.index].head(n)

@st.cache()
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

qa, summ = load_models()
paragraphs, paragraphs_embedded = load_data()

q = st.text_input('What is your query?')
if q:
    ans = ask(q, X=paragraphs_embedded, s=paragraphs, n=3, model=qa)
    for i,t in ans.values:
        with st.beta_expander(f'PAGE {i}'):
            if len(t)>45:
                summary = summarize(t, summ, 1)
                st.success(summary)
                st.write(bold_sentences(t,summary))
            else:
                st.write(t)