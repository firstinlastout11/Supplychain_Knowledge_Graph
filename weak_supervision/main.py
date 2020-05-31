import pandas as pd
import numpy as np
from spacy.matcher import Matcher 
import spacy
import snorkel
from sklearn.model_selection import train_test_split
from snorkel.labeling.model import LabelModel

from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.analysis import metric_score
from snorkel.utils import probs_to_preds
from snorkel.labeling import filter_unlabeled_dataframe

from utils import *
from LF import *

def string_to_tuple(df):
    
    tups = []
    
    for idx, row in df.iterrows():
        temp = []

        for tok in row['position'].split(", "):
            num = int(tok.replace("(", "").replace(")", "")) 
            temp.append(num) 
        
        tups.append(temp)
    return tups

def load_data():
    
    # load data
    df = pd.read_csv("final_merged.csv")
    df = df[['Column','source','target_x','supply','sentence','position']]
    df = df.rename(columns={'target_x':'target'})
    df = df[df['position'].notnull()]
    
    # change string into tuples for the position value 
    tups_lst = string_to_tuple(df)
    df['position'] = tups_lst

    # replace string to numbers
    df['supply'] = df['supply'].replace('0',0)
    df['supply'] = df['supply'].replace('0.0',0)
    df['supply'] = df['supply'].replace('1.0',1)
    df['supply'] = df['supply'].replace('1',1)
    df['supply'] = df['supply'].replace('?',0)
    
    return df

def data_preprocess(df):

    # Initiate new lists to store the pre-processed values
    tokens_lst = []
    left_tokens_lst = []
    right_tokens_lst = []

    # Data pre-processing
    for idx, row in df.iterrows():

        # change the sentence into spacy's object
        doc = nlp(row['sentence'])

        # token list of the sentence        
        toks = [tok.orth_ for tok in doc]

        # store position
        start, end = row['position']

        # append the values to the lists
        tokens_lst.append(toks)
        left_tokens_lst.append(toks[:start])
        right_tokens_lst.append(toks[end+1:])

    # Assign those computed lists into the datafamr
    df['tokens'] = tokens_lst
    df['left_tokens'] = left_tokens_lst
    df['right_tokens'] = right_tokens_lst

    # split the dataframe based on the labels: 1, 0, unknown
    df_zero = df[df['supply'] == 0]
    df_one = df[df['supply'] == 1]
    df_null = df[df['supply'].isnull()]

    # types into integers
    df_zero['supply'] = df_zero['supply'].astype('int64')
    df_one['supply'] = df_one['supply'].astype('int64')

    # unlabeled data become training set
    df_train = df_null[['source','target','sentence','position','tokens','left_tokens','right_tokens']]

    # creating the dataframes
    X_one = df_one[['source','target','sentence','position','tokens','left_tokens','right_tokens']]
    Y_one = np.array(df_one['supply'])


    X_zero = df_zero[['source','target','sentence','position','tokens','left_tokens','right_tokens']]
    Y_zero = np.array(df_zero['supply'])


    # split the labeled dataframe into dev and test set
    X_one_val, X_one_test, Y_one_val, Y_one_test = train_test_split(X_one, Y_one, test_size = 0.5)
    X_zero_val, X_zero_test, Y_zero_val, Y_zero_test = train_test_split(X_zero, Y_zero, test_size = 0.5)

    # concatenate the 1, 0 labeled data
    df_dev = pd.concat([X_one_val,X_zero_val])
    Y_dev = np.append(Y_one_val,Y_zero_val)

    df_test = pd.concat([X_one_test,X_zero_test])
    Y_test = np.append(Y_one_test,Y_zero_test)

    return df_dev, Y_dev, df_train, df_test, Y_test


def label_model_creator(df_dev, Y_dev, df_train, df_test, Y_test):

     # Accumulate all the labeling_functions for supply
    supply_lfs = [
        lf_supply,
        lf_customer,
        lf_sales_to,
        lf_our_customer,
        lf_acquisition,
        lf_people,
        lf_sold,
        lf_relation,
        lf_competition
    ]

    # Apply the above labeling functions to the data in Pandas dataframe formats
    applier = PandasLFApplier(supply_lfs)

    # Use the applier of the labeling functions to both development set and train set
    L_dev = applier.apply(df_dev)
    L_train = applier.apply(df_train)
    L_test = applier.apply(df_test)


    # caridnality : 2 (True and False)
    label_model = LabelModel(cardinality=2, verbose=True)

    # Fit the label_model
    label_model.fit(L_train, Y_dev, n_epochs=5000, log_freq=500)

    # accuracy for the label model using the test set
    label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        "accuracy"
    ]
    print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
        
    # check the F-1 score and ROC_AUC score
    probs_dev = label_model.predict_proba(L_dev)
    preds_dev = probs_to_preds(probs_dev)
    print(
        f"Label model f1 score: {metric_score(Y_dev, preds_dev, probs=probs_dev, metric='f1')}"
    )
    print(
        f"Label model roc-auc: {metric_score(Y_dev, preds_dev, probs=probs_dev, metric='roc_auc')}"
    )

    return label_model, L_train

def label_model_trainer(label_model, L_train, df_train):

    """
    To train the extraction model, 
    we first output the probabilities of the binary choices: True and False from our label model.
    Then, using the probabilities, we train our end model
    """

    # extract the probabiliteis from the training set using our label model
    probs_train = label_model.predict_proba(L_train)

    # Since we cannot use the data points that did not receive any labels (Not covered by our labeling functions),
    # we filter them out

    # extract only the data points that received any labels from the labeling functions
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df_train, y=probs_train, L=L_train
    )

    X_train = uniform_length(df_train_filtered)
    model = rnn_model()
    batch_size = 64
    model.fit(X_train, probs_train_filtered, batch_size=batch_size, epochs=50)

    X_test = uniform_length(df_test)
    probs_test = model.predict(X_test)
    preds_test = probs_to_preds(probs_test)
    print(
        f"Test F1 when trained with soft labels: {metric_score(Y_test, preds=preds_test, metric='f1')}"
    )
    print(
        f"Test ROC-AUC when trained with soft labels: {metric_score(Y_test, probs=probs_test, metric='roc_auc')}"
    )

if __name__ == "__main__":
    
    # load data
    df = load_data()

    # pre-process the data
    df_dev, Y_dev, df_train, df_test, Y_test = data_preprocess(df)

    # get the trained label model
    label_model, L_train = label_model_creator(df_dev, Y_dev, df_train, df_test, Y_test)

    # train the label-model end model
    label_model_trainer(label_model, L_train, df_train)

    
