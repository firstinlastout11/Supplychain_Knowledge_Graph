#from google.colab import drive
import pandas as pd
import numpy as np
import spacy, snorkel, torch
from sklearn.model_selection import train_test_split
from torchtext import data, vocab
from supply_classifier import *
import torch.optim as optim
from utils import train, evaluate


def text_preprocessing():
    """
        This text pre-processing steps require pytorch's text module called torch text.<br>
        It can simply be installed using the command "pip install torchtext"<br><br>
        Our inputs contain two types of text: left_text, right_text based on the location of the target entity.
        We instantiate our data objects using torchtext as below.

        Both left and right sentences are tokenized by word by spaCy's tokenizer <br>
        For the embedding of the text, we ustilze the glove's 300-d pretrained word-vectors. It can be downloaded easily online. Or, it can be replaced by any word-vectors.

    """
    # Define our Torch Data objects
    TEXT = data.Field(tokenize = "spacy", batch_first = True, include_lengths = True)

    LABEL = data.Field(dtype = torch.float, sequential = False, use_vocab=False, batch_first = True)

    # Define the fields to be used later for data-processing
    fields = [(None, None), ('left_text',TEXT), ('right_text',TEXT), ('label',LABEL)]

    # we use the dataset that contains the columns (idx, left_sentence, right_sentence, supply_label)
    # we make the dataset into torchtext's Dataset object using the fields defined above
    #training_data = data.TabularDataset(path = 'drive/My Drive/Data/partitioned_sents_final.csv', format='csv', fields=fields, skip_header=True)
    training_data = data.TabularDataset(path = 'partitioned_sents_final.csv', format='csv', fields=fields, skip_header=True)

    # We split the training and valid dataset (We use 80-20 ratio)
    train_data, valid_data = training_data.split(split_ratio=0.8)

    # For each sentence, we want to embed them using the pre-trained Glove vector (300-dimension)
    #embeddings = vocab.Vectors('glove.6B.300d.txt', 'drive/My Drive/Data/')
    embeddings = vocab.Vectors('glove.6B.300d.txt', 'glove.6B/')

    # Build the vocab based on the Glove vector. Min_freq:3, Max_size=5,000
    TEXT.build_vocab(train_data, min_freq = 3, max_size = 5000, vectors = embeddings)
    LABEL.build_vocab(train_data)

    # Store the vocab size. Note that the vocab contains extra 2 words ( <UNKNOWN>, <PAD>))
    vocab_size = len(TEXT.vocab)

    return train_data, valid_data, TEXT, LABEL

def model_instantiate(TEXT, LABEL):

    # Hyper-parameters
    vocab_size = len(TEXT.vocab)
    embedding_dim = 300
    hidden_nodes = 32
    output_nodes = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.2

    # GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate the end-model
    model = SupplyClassifier(vocab_size = vocab_size,
                            embedding_dim = embedding_dim,
                            hidden_dim = hidden_nodes,
                            output_dim = output_nodes,
                            n_layers = num_layers,
                            bidirectional = bidirectional,
                            dropout = dropout)

    # use GPU if available
    model = model.to(device)

    # Store the pre-trained embeddings for each word and input into our model
    gloves = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(gloves)

    # Initialize the pretrained embedding
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Define the optimizer. We use ADAM.
    optimizer = optim.Adam(model.parameters())

    # Define the loss function. We use BCELoss since it is a binary classification
    loss_criterion = nn.BCELoss()
    loss_criterion = loss_criterion.to(device)

    return model, optimizer, loss_criterion

def iterators():

    # Define the batch size
    BATCH_SIZE = 64
    
    # GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the train and valid iterator using BucketIterator
    train_iterator, valid_iterator = data.BucketIterator.splits(
            (train_data, valid_data),
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.left_text), # Sort the batches by text length size
            sort_within_batch = True,
            device = device)

    return train_iterator, valid_iterator

if __name__ == "__main__":
    
    # load spacy's nlp object
    nlp = spacy.load('en')

    # Load the header file for motor industry
    train_data, valid_data, TEXT, LABEL = text_preprocessing()

    # Instantaite the model
    model, optimizer, loss_criterion = model_instantiate(TEXT, LABEL)

    # get iterators
    train_iterator, valid_iterator = iterators

    # number of epochs
    N_EPOCHS = 30
    best_val_loss = float('inf')

    PATH = 'saved_weights.pt'
    for epoch in range(N_EPOCHS):
        
        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, loss_criterion)
        
        # evaluate the model
        val_loss, val_acc = evaluate(model, valid_iterator, loss_criterion)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATH)
            
        print(f'Epoch {epoch + 1} Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'Epoch {epoch + 1} Validation Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')

