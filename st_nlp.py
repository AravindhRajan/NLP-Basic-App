# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:46:28 2020

@author: Aravindh Rajan
"""

import pandas as pd
import numpy as np
import pickle
import os
import random
#import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import metrics
#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.metrics import confusion_matrix
#from sklearn.cluster import DBSCAN
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import normalize 
#from sklearn.decomposition import PCA 
#from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
import pydeck as pdk 
import altair as alt 
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.collocations import *
import spacy
import string

import heapq
import re

st.write("""
# Natural Language Processing
### This application is used to perform rudimentary NLP tasks.
""")


action = st.selectbox('What do you want to perform?',['Select','Word Tokenization',
                                                      'Sentence Tokenization','SW Removal',
                                                      'Identify Bigrams','POS Tagging',
                                                      'Text similarity','Text Summarization'])

if action == 'Text similarity':
    text = st.text_input('Input your sentence here:',key="x") 
    text2 = st.text_input('Input your sentence here:',key="y") 
elif action != 'Text similarity':
    text = st.text_input('Input your sentence here:',key="x")

#words = word_tokenize(text)
#customSW = set(stopwords.words('english')+list(punctuation))
#wordsWOsw = [word for word in words if word not in customSW]
    

if st.button('Run'):
    if action == 'Sentence Tokenization':
        sents = sent_tokenize(text)
        st.write(action)
        st.write(sents)
    elif action == 'Word Tokenization':
        words = word_tokenize(text)
        #words = [word_tokenize(sent) for sent in sents]
        st.write(action)
        st.write(words)
    elif action == 'SW Removal':
        words = word_tokenize(text)
        customSW = set(stopwords.words('english')+list(punctuation))
        wordsWOsw = [word for word in words if word not in list(customSW)]
        st.write('Tokenization')
        st.write(words)
        st.write('Stop words removed')
        st.write(wordsWOsw)
        #elif action == 'Identify Bigrams':
        #    sents = sent_tokenize(text)
        #    words = [word_tokenize(sent) for sent in sents]
        #    customSW = set(stopwords.words('english')+list(punctuation))
        #    wordsWOsw = [word for word in words if word not in list(customSW)]
        #    bigram_measures = nltk.collocations.BigramAssocMeasures()
        #    finder = BigramCollocationFinder.from_words(wordsWOsw)
        #    st.write(finder.ngram_fd.items())
    elif action == 'POS Tagging':
        words = word_tokenize(text)
        customSW = set(stopwords.words('english')+list(punctuation))
        wordsWOsw = [word for word in words if word not in list(customSW)]
        wordsSWremoved = (" ".join(wordsWOsw))
        out_lst = nltk.pos_tag(word_tokenize(wordsSWremoved))
        out_df = pd.DataFrame(out_lst[0:]).rename(columns={0:'Word',1:'POS'})
        st.write(out_df)
    elif action == 'Text similarity':
        sents = [text,text2]
        
        def clean_str(text):
            text = word_tokenize(text)
            sent_without_sw = [word for word in text if not word in stopwords.words()]
            sent_without_punc = ' '.join([word for word in sent_without_sw if not word in punctuation])
            return sent_without_punc
    
        cleaned = list(map(clean_str,sents))
        
        vectorizer = CountVectorizer().fit_transform(cleaned)
        vectors = vectorizer.toarray()
        
        csim = cosine_similarity(vectors)
        
        def cosine_sim(vec1,vec2):
            vec1 = vec1.reshape(1,-1)
            vec2 = vec2.reshape(1,-1)
            return cosine_similarity(vec1,vec2)[0][0]
    
        st.write(cosine_sim(vectors[0],vectors[1]))
    elif action == 'Text Summarization':
        article_text = re.sub(r'\[[0-9]*\]', ' ', text)
        article_text = re.sub(r'\s+', ' ', article_text)
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        sentlst = nltk.sent_tokenize(article_text)
        # finding weighted freq of occurence
        word_frequencies = {}                    
        text2 = word_tokenize(formatted_article_text)
        text3 = [word for word in text2 if not word in stopwords.words()]
        for word in text3:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        
        maximum_frequncy = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        # sentence scores
        sentence_scores = {}
        for sent in sentlst:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        # summarize
        summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)
        st.write('Summarized text')
        st.write(summary)
        
    
