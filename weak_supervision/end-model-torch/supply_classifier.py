import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class SupplyClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        # Define the Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the LSTM layer for each of the text using the hyparameters defined
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional,
                           dropout = dropout, batch_first = True)

        # Define the dense layer
        # Note the input dimension : 2*2*hidden_dim because we concat left and right sentences which is Bi-drectional
        self.fc = nn.Linear(2 * 2 * hidden_dim, output_dim)

        # Define the activation function. Since it is a binary-classification, we use sigmoid
        self.act = nn.Sigmoid()

    def forward(self, left_text, right_text, left_text_lengths, right_text_lengths):
        
        # embedded vectors for the left and right text
        left_embedded = self.embedding(left_text)
        right_embedded = self.embedding(right_text)
        
        """ We use the pack_padded_sequence to make all the sentence unform in the length
            Sentences that are shorter than text_lengths are filled with <pad>
            Note that <pad> is not used as a part of inputs to the LSTM model
            By setting batch_first = True, the inputs have the shape [batch size, sentence length, embedding dim]
        """
        
        left_packed_emb = pack_padded_sequence(left_embedded, left_text_lengths, batch_first = True, enforce_sorted=False)
        right_packed_emb = pack_padded_sequence(right_embedded, right_text_lengths, batch_first = True, enforce_sorted=False)
        
        
        # we store the outputs, hidden_states and cell_states for each of the sentence
        left_out, (l_hidden, l_cell) = self.lstm(left_packed_emb)
        right_out, (r_hidden, r_cell) = self.lstm(right_packed_emb)
        
        # Since our model is Bi-LSTM, we need to concatenate the hidden states for each direction
        l_hidden = torch.cat((l_hidden[-2,:,:], l_hidden[-1,:,:]), dim = 1)
        r_hidden = torch.cat((r_hidden[-2,:,:], r_hidden[-1,:,:]), dim = 1)
        
        # Finally, we concatenate the hidden states for both left and right sentence
        hidden = torch.cat((l_hidden, r_hidden), dim = 1)
        
        # We input the concatenated hidden states into out Fully-connected layer
        fc_out = self.fc(hidden)
        
        # Then, we acquire the final results by putting the fc_out into our sigmoid activation function
        final = self.act(fc_out)
        
        return final
        