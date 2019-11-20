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


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
def train():
    """
    Get dataset and parameters
    """
    params = parameters()
    input_set, target_set, seq_len_set, original_set = load_data(params)
    input_set, _ = pad_sequences(input_set, dtype=np.float32)
    target_set, _ = pad_sequences(target_set, dtype=np.int64)
    params.d_model = input_set.shape[-1]
    params.target_vocab_size = len(params.dictionary)
    """
        Define loss, model, optimizer
    """
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')


    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)