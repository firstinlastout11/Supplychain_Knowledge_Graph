from pathlib import Path
import spacy 
import en_core_web_sm
import re 
import string 
import pandas as pd 
import numpy as np 
import math 

from spacy.matcher import Matcher 
from spacy.tokens import Span 


def supply(text,company, nlp):
    doc = nlp(text)
    org_ents = []
    dic = {}
    ents_cand = []
    sents = []

    for sent in doc.sents:
        if "supply" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    for sent in doc.sents:
        if "suppli" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    ents_cand = [str(s) for s in ents_cand]
    
    source = [company]*len(ents_cand)
    relation = ['Supply']*len(ents_cand)
    target = []
    for i in range(len(ents_cand)):
        target.append(str(ents_cand[i]))
    df = pd.DataFrame({'source':source,'target':target,'relation':relation,'sentence':sents})
    df = df.drop_duplicates()
        
        
    return df   

def customers(text, company, nlp):
    doc = nlp(text)
    org_ents = []
    dic = {}
    ents_cand = []
    sents = []

    for sent in doc.sents:
        if "customer" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    ents_cand = [str(s) for s in ents_cand]
    
    source = [company]*len(ents_cand)
    relation = ['customers']*len(ents_cand)
    target = []
    for i in range(len(ents_cand)):
        target.append(str(ents_cand[i]))
    df = pd.DataFrame({'source':source,'target':target,'relation':relation,'sentence':sents})
    df = df.drop_duplicates()
        
        
    return df

def sales_to(text, company, nlp):
    doc = nlp(text)
    org_ents = []
    dic = {}
    ents_cand = []
    sents =[]

    for sent in doc.sents:
        if "sales to" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    ents_cand = [str(s) for s in ents_cand]
    
    source = [company]*len(ents_cand)
    relation = ['Sells to']*len(ents_cand)
    target = []
    for i in range(len(ents_cand)):
        target.append(str(ents_cand[i]))
    df = pd.DataFrame({'source':source,'target':target,'relation':relation, 'sentence':sents})
    df = df.drop_duplicates()
    return df

def dependent(text,company, nlp):
    doc = nlp(text)
    org_ents = []
    dic = {}
    ents_cand = []
    sents = []

    for sent in doc.sents:
        if "dependent on" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    for sent in doc.sents:
        if "depends on" in sent.text.lower():
            doc = nlp(sent.text)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and company.lower() not in ent.text.lower():
                    ents_cand.append(ent)
                    sents.append(sent.text)

    ents_cand = [str(s) for s in ents_cand]
    
    source = [company]*len(ents_cand)
    relation = ['dependent on']*len(ents_cand)
    target = []
    for i in range(len(ents_cand)):
        target.append(str(ents_cand[i]))
    df = pd.DataFrame({'source':source,'target':target,'relation':relation, 'sentence':sents})
    df = df.drop_duplicates()
        
        
    return df