from transformer import *
from utils import *
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

class parameters():

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = 0
    target_vocab_size = 0
    dropout_rate = 0.1