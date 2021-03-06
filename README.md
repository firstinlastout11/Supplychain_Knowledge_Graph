# Supply chain knowledge graph

## Overview
In this project, we want to construct a knowledge graph related to suppy-chain relations among the companies. 

## Knowledge graph
Knowledge graph is a form of database to store relations among entities in the database. It consists of three primary components: Source, Target and Edge. Source and Target each represent a specific entity and Edge represents a relation between the two entities. 
In this project, both Source and Target wiill present a speicific company and Edge will represent a supply-chain relation bewteen the two companies.

## Data
For this project, the primary source of data is the EDGAR database run by the U.S. Securities and Exchange Commission (SEC). 
The database contains various financial filings (10-Q and 10-K) each company is obligated to fill and submit both quarterly and yearly. 

For the scope of the project, we are currently utilizing 10-K files for each company but we will expand to different files such as 10-Q and etc.

All the filings are in the form of text so they need to be processed accordingly.

Considering the size of the datasets, some of the datasets are unavailable to load in the repository. I can provide them upon request.

link: https://www.sec.gov/edgar.shtml

## Files
* `weak_supervision/end-model-torch/main.py` contains python code to instantiate, train and evaluate the Bi-LSTM text-classification model in Pytorch

* `weak_supervision/end-model-torch/End_Model.ipynb` is a notebook that describes thet steps to create the final end-model descrbied above.

* `weak_supervision/main.py` contains python code to perform the knowledge-graph creation using weak-supervision: 1) Extracting 10-K html file of each company, 2) Pre-process and normalize the text, 3) Extract supply-chain relations using the labeling functions

* `weak_supervision/utils.py` contains python code to perform the knowledge-graph creation using weak-supervision: 1) Extracting 10-K html file of each company, 2) Pre-process and normalize the text, 3) Extract supply-chain relations using the labeling functions

* `weak_supervision/LF.py` contains python code to perform the knowledge-graph creation using weak-supervision: 1) Extracting 10-K html file of each company, 2) Pre-process and normalize the text, 3) Extract supply-chain relations using the labeling functions

* `weak_supervision/Weak_Sueprvision` contains python code to perform the knowledge-graph creation using weak-supervision: 1) Extracting 10-K html file of each company, 2) Pre-process and normalize the text, 3) Extract supply-chain relations using the labeling functions

* `keyword_rel_extract/main.py` contains python code to perform the knowledge-graph creation process: 1) Extracting 10-K html file of each company, 2) Pre-process and normalize the text, 3) Extract supply-chain relations using the labeling functions

* `keyword_rel_extract/lab_functions.py` contains python code to perform the information extraction process by utilizing Named Entity Recognition and Rule-based learning

* `header_scrape/header_scrape.ipynb` jupyter notebook conatains a sample process to extract header files of each company

* `Network_drawing.ipynb` jupyter notebook performs a plotting of the result and save the file as a knowledge-graph drawing using networkx


## How to run the codes

Since the language used in the project is Python3, it needs to be installed. Also, it utilizes various external libraries. Some of the primary packages are as below:

* SpaCy : A natural language processing framework. https://spacy.io/usage
* Snorkel : A data-programming framework. https://www.snorkel.org/
* Pytorch : A deep learning framework
* Keras : A deep learning framework
* NLTK : A natural language toolkit. https://www.nltk.org/api/nltk.html
* BeatifulSoup : A web-scraping library 



With all the above ready, on your command line,
Execute `python3 main.py` 

## Sample result
Below is the image of the sample network created by the project
![Knowledge_graph](/images/automobile_network.png)
