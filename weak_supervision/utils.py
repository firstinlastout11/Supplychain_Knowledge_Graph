import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTM,
)
import numpy as np
from typing import Tuple

#def uniform_length(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
def uniform_length(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

    """ 
        since the length of sentence varies much, we make the lengths uniform
    """
    
    # extract three: tokens, left_tokens, right_tokens
    toks = df.tokens
    left_toks = df.left_tokens
    right_toks = df.right_tokens
    
    
    def token_filter(l, max_len=50):
        return l[:max_len] + [""] * (max_len - len(l))

    tokens = np.array(list(map(token_filter, toks)))
    left_tokens = np.array(list(map(token_filter, left_toks)))
    right_tokens = np.array(list(map(token_filter, right_toks)))
    
    return left_tokens, right_tokens
    #return tokens, left_tokens, right_tokens

def bidirectional_lstm(tokens: tf.Tensor, rnn_state_size: int = 64, num_buckets: int = 40000, embed_dim: int = 36,):
    """
        Bidirectional LSTM model
    """
    
    # Converts each string in the input Tensor to its hash mod by a number of buckets.
    ids = tf.strings.to_hash_bucket_fast(tokens, num_buckets)
    
    # Turns positive integers (indexes) into dense vectors of fixed size
    embedded_input = Embedding(num_buckets, embed_dim)(ids)
    
    # return the bidrecitonal LSTM
    return Bidirectional(LSTM(rnn_state_size, activation=tf.nn.relu))(
        embedded_input, mask=tf.strings.length(tokens)
    )


def rnn_model(
    rnn_state_size: int = 64, num_buckets: int = 40000, embed_dim: int = 12) -> tf.keras.Model:
    
    """
    This lstm model predicts the label probailities given the embedded tokens
    
    rnn_state_size: state size of LSTM model
    num_buckets: Number of buckets to hash strings to integers
    embed_dim: Size of token embeddings

    """
    #toks_ph = Input((None,), dtype="string")
    #toks_embs = bidirectional_lstm(b_ph, rnn_state_size, num_buckets, embed_dim
    #layer = Concatenate(1)([left_embs, bet_embs, right_embs])
    
    # Instantiate Input Keras Object. Data type : string
    left_obj = Input((None,), dtype="string")
    right_obj = Input((None,), dtype="string")
    
    # intput embeddings
    left_lstm = bidirectional_lstm(left_obj, rnn_state_size, num_buckets, embed_dim)
    right_lstm = bidirectional_lstm(right_obj, rnn_state_size, num_buckets, embed_dim)
    
    # concatenate two inputs
    layer = Concatenate(1)([left_lstm, right_lstm])
    
    # Dense layers with relu activations
    layer = Dense(64, activation=tf.nn.relu)(layer)
    layer = Dense(32, activation=tf.nn.relu)(layer)
    
    # Output layer with softmax activation
    probabilities = Dense(2, activation=tf.nn.softmax)(layer)
    
    #  final model using the characteristics above
    model = tf.keras.Model(inputs=[left_obj, right_obj], outputs=probabilities)
    
    #model = tf.keras.Model(inputs=[bet_ph, left_ph, right_ph], outputs=probabilities)
    
    # compile the model: AdagradOptimizer, cross_entropy
    model.compile(tf.train.AdagradOptimizer(0.1), "categorical_crossentropy")
    return model
