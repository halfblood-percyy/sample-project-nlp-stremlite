#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Necessary imports
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import nltk


# In[2]:


#Headings for Web Application
st.title("Natural Language Processing Web Application Example")
st.subheader("What type of NLP service would you like to use?")


# In[3]:


#Picking what NLP task you want to do
option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization')) #option is stored in this variable


# In[4]:


#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze.")
text = st.text_input('Enter text') #text is stored in this variable


# In[5]:


#Display results of the NLP task
st.header("Results")


# In[6]:


#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList


# In[9]:


#Sentiment Analysis
if option == 'Sentiment Analysis':

    #Creating graph for sentiment across each sentence in the text inputted
    nltk.download('punkt')
    sents = sent_tokenize(text)
    entireText = TextBlob(text)
    sentScores = []
    for sent in sents:
        text = TextBlob(sent)
        score = text.sentiment[0]
        sentScores.append(score)

    #Plotting sentiment scores per sentencein line graph
    st.line_chart(sentScores)

    #Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    st.write("The sentiment of the overall text below.")
    st.write(sentimentTotal)
    
#Named Entity Recognition
elif option == 'Entity Extraction':
    #Getting Entity and type of Entity
    entities = []
    entityLabels = []
    doc = nlp(text)
    for ent in doc.ents:
        entities.append(ent.text)
        entityLabels.append(ent.label_)
    entDict = dict(zip(entities, entityLabels)) #Creating dictionary with entity and entity types
    
    #Using function to create lists of entities of each type
    entOrg = entRecognizer(entDict, "ORG")
    entCardinal = entRecognizer(entDict, "CARDINAL")
    entPerson = entRecognizer(entDict, "PERSON")
    entDate = entRecognizer(entDict, "DATE")
    entGPE = entRecognizer(entDict, "GPE")

    #Displaying entities of each type
    st.write("Organization Entities: " + str(entOrg))
    st.write("Cardinal Entities: " + str(entCardinal))
    st.write("Personal Entities: " + str(entPerson))
    st.write("Date Entities: " + str(entDate))
    st.write("GPE Entities: " + str(entGPE))
    
#Text Summarization
else:
    from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
    from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
    from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

    document = text
    summary =""

    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = [".", "/n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Summarize document.
    result_dict = auto_abstractor.summarize(document, abstractable_doc)
    for sentence in result_dict["summarize_result"]:
        summary +=sentence
        
    summWords = summary
    st.subheader("Summary")
    st.write(summWords)


# In[ ]:




