import re
import requests
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from spacy import displacy
from pathlib import Path
from lab_functions import *
from time import sleep 



def restore_windows_1252_characters(restore_string):
    """
        Replace C1 control characters in the Unicode string s by the
        characters at the corresponding code points in Windows-1252,
        where possible.
    """

    def to_windows_1252(match):
        try:
            return bytes([ord(match.group(0))]).decode('windows-1252')
        except UnicodeDecodeError:
            # No character at the corresponding code point: remove it.
            return ''
        
    return re.sub(r'[\u0080-\u0099]', to_windows_1252, restore_string)

def entry_creation(motor_df):


    comp_cik = motor_df[['CENTRAL INDEX KEY','COMPANY CONFORMED NAME']]
    comp_cik = comp_cik.rename(columns={'CENTRAL INDEX KEY':'cik', 'COMPANY CONFORMED NAME':'name'})

    # Store the CIKs for the motor vehicle industry
    cik_lst = list(motor_df['CENTRAL INDEX KEY'])
    #cik_lst = ['355811']


    # Initialize a dictionary to store the information for the filings
    entry_dict = {}



    # Loop through all the companies
    for cik in cik_lst:
        # base URL for the SEC EDGAR browser
        endpoint = r"https://www.sec.gov/cgi-bin/browse-edgar"

        # define our parameters dictionary
        param_dict = {'action':'getcompany',
                    'CIK': cik,
                    'type':'10-k',
                    'owner':'exclude',
                    'output':'atom',
                    'count':'10'}

        # request the url, and then parse the response.
        response = requests.get(url = endpoint, params = param_dict)
        soup = BeautifulSoup(response.content, 'lxml')

        # find all the entry tags
        entries = soup.find_all('entry')

        #  initalize theh dictionary
        entry_dict[cik] = {}

        # loop through each found entry, store the information
        for entry in entries[:1]:

            # grab the accession number
            accession_num = entry.find('accession-nunber').text

            # category info
            category_info = entry.find('category')    
            label = category_info['term']


            # create a new dictionary
            entry_dict[cik][accession_num] = {}
            entry_dict[cik][accession_num]['date'] = entry.find('filing-date').text
            entry_dict[cik][accession_num]['type'] = label
            entry_dict[cik][accession_num]['link'] = entry.find('link')['href']
            entry_dict[cik][accession_num]['text'] = entry.find('link')['href'].replace('-index.htm','.txt')
            entry_dict[cik][accession_num]['name'] = soup.find('conformed-name').text
        
    return entry_dict

def relation_extraction(entry_dict, cik_lst):
    labeling_functions = [supply, customers, sales_to, dependent]
    df = pd.DataFrame()

    nlp = spacy.load("en_core_web_md")
    nlp.max_length = 1200000 

    for cik in cik_lst:
        for entry in entry_dict[cik]:

            # define the url to specific html_text file
            new_html_text = entry_dict[cik][entry]['text']
            
            # company name
            comp_name = entry_dict[cik][entry]['name']
            
            # grab the response
            response = requests.get(new_html_text)

            # pass it through the parser, in this case let's just use lxml because the tags seem to follow xml.
            soup = BeautifulSoup(response.content, 'lxml')

            master_document_dict = {}
            filing_documents = {}

            for filing_document in soup.find_all('document'):

                document_id = filing_document.type.find(text=True, recursive=False).strip()
                if document_id == '10-K':
                    master_document_dict[document_id] = {}

                    filing_doc_text = filing_document.find('text').extract()
                    filing_doc_string = str(filing_doc_text)


                    repaired_pages = {}
                    normalized_text = {}

                    doc_soup = BeautifulSoup(filing_doc_string, features="html5lib")
                    doc_text = doc_soup.html.body.get_text(' ', strip=True)
                    doc_text_norm = restore_windows_1252_characters(unicodedata.normalize('NFKD',doc_text))
                    doc_text_norm = doc_text_norm.replace(' ', ' ').replace('\n',' ')

                    filing_documents[document_id] = doc_text_norm


                    doc = nlp(filing_documents['10-K'])
                    sentences_full = [x for x in doc.sents]
                    sentences_full = "".join(map(str, sentences_full))

                    for lab_func in labeling_functions:
                        temp = lab_func(filing_documents['10-K'], comp_name, nlp)

                        df = pd.concat([df,temp])
                        #displacy.render(nlp(str(sentences_full)), jupyter=True, style='ent')
    return df


def  entity_linking(df):

    headers = {'accept':'text/html'}
    relation_df = pd.DataFrame()
    # Base URL for Spotlight API
    base_url = "http://api.dbpedia-spotlight.org/en/annotate"

    for comp in df['target']:
        params = {'text':[comp], 'confidence':0.35}
        res - requests.get(base_url, params = params, headers=headers)
        sleep(0.5)

        if res.text.find('httpL//dbpedia.org') == -1:
            relation_df = relation_df.append(row)
    
    
    

if __name__ == "__main__":
    


    # Load the header file for motor industry
    motor_df = pd.read_csv('data/motor_header.csv')

    # Store the CIKs for the motor vehicle industry
    cik_lst = list(motor_df['CENTRAL INDEX KEY'])
    #cik_lst = ['355811']

    entry_dict = entry_creation(motor_df)
    result_df = relation_extraction(entry_dict, cik_lst)
    result_df.to_csv('result_df_exp.csv')
