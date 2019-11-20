from transformer import *
from utils import *
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

class parameters():
    number_sentence = 3
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = 0
    target_vocab_size = 0
    dropout_rate = 0.1
    freq = 100
    feature = 1
    data = 'eeg'

def train()
    params = parameters()
    input_set, target_set, seq_len_set, original_set = load_data(params)
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)