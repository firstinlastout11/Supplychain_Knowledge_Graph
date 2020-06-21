#from google.colab import drive
import torch
from torchtext import data, vocab
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

def binary_accuracy(preds, y):
    """
        This function computes the binary accuracy.
    """
    # round the preds to 0 or 1
    preds = torch.round(preds)
    
    # check if the preds are correct
    check = (preds == y).float()
    
    # accuracy
    acc = check.sum() / len(check)
    
    return acc


def train(model, iterator, optimizer, loss_criterion):
    """
        This model trains the end model given the iterator, optimizer and loss_criterion
    """
    # set the accuracy and loss to zero every iteration
    loss_epoch = 0
    acc_epoch = 0
    
    # Initialize the training phase
    model.train()
    for batch in iterator:
        
        # zero out the gradients
        optimizer.zero_grad()
        
        # for each batch, store the text and the length of the sentences
        left_text, left_text_len = batch.left_text
        right_text, right_text_len = batch.right_text

        # flatten to 1-D
        preds = model(left_text, right_text, left_text_len, right_text_len).squeeze()

        # compute the loss of the batch
        loss = loss_criterion(preds, batch.label.squeeze()) 

        # compute the accuracy for the batch
        acc = binary_accuracy(preds, batch.label)

        # perform back-prop
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # accumulate the loss and accuracy
        loss_epoch += loss.item()
        acc_epoch += acc.item()
    return loss_epoch / len(iterator), acc_epoch / len(iterator)

def evaluate(model, iterator, loss_criterion):

    # set the accuracy and loss to zero every iteration
    loss_epoch = 0
    acc_epoch = 0
    
    #Initialize the evaluation phase
    model.eval()
    
    # We don't need to record the grads for this evaluation process
    with torch.no_grad():
        
        for batch in iterator:
            
            # for each batch, store the text and the length of the sentences
            left_text, left_text_len = batch.left_text
            right_text, right_text_len = batch.right_text            
            # flatten to 1-D
            preds = model(left_text, right_text, left_text_len, right_text_len).squeeze(1)

            # compute the loss of the batch
            loss = loss_criterion(preds, batch.label)
        
            # compute the accuracy for the batch
            acc = binary_accuracy(preds, batch.label)

            # accumulate the loss and accuracy
            loss_epoch += loss.item()
            acc_epoch += acc.item()
            
    return loss_epoch / len(iterator), acc_epoch / len(iterator)

def predict(model, df, nlp, TEXT, device):
    """
        This function takes a dataframe and outputs the predicted values of each row
    """
    result = []
    probs_result = []
    for idx, row in df.iterrows():
        left_sent = row['left_sents']
        right_sent = row['right_sents']

        left_tokenized = [tok.text for tok in nlp.tokenizer(left_sent)]
        right_tokenized = [tok.text for tok in nlp.tokenizer(right_sent)]

        left_indexed = [TEXT.vocab.stoi[t] for t in left_tokenized]
        right_indexed = [TEXT.vocab.stoi[t] for t in right_tokenized]

        l_length = [len(left_indexed)] 
        r_length = [len(right_indexed)] 

        l_tensor = torch.LongTensor(left_indexed).to(device)              #convert to tensor
        l_tensor = l_tensor.unsqueeze(1).T 

        r_tensor = torch.LongTensor(right_indexed).to(device)
        r_tensor = r_tensor.unsqueeze(1).T

        l_len = torch.LongTensor(l_length)
        r_len = torch.LongTensor(r_length)

        prediction = model(l_tensor, r_tensor, l_len, r_len)
        probs_result.append([1 - prediction.item(), prediction.item()])
        result.append(round(prediction.item()))
    return result, probs_result