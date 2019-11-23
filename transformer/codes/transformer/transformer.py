import tensorflow as tf
import numpy as np
from attention import *
from positional_encoding import *
from utils import *


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers,d_model, num_heads, dff,pe_max_len
                 , rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = positional_encoding(pe_max_len, self.d_model)

        self.input_proj = tf.keras.models.Sequential(name='en_proj')
        self.input_proj.add(tf.keras.layers.Dense(units=self.d_model,kernel_initializer='glorot_normal'))
        # self.input_proj.add(tf.keras.layers.Dropout(rate=dp))
        self.input_proj.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))

        self.dropout = tf.keras.layers.Dropout(rate=0.1, name='en_proj_dp')
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = tf.reshape(x, [x.shape[0], x.shape[1], -1])  # B*T*(D*n)  flatten on channels
        # adding embedding and position encoding.
        x = self.input_proj(x)  # (batch_size, input_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_target,rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        self.pre_net = Pre_Net()
    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):

        inp = self.pre_net(inp, training)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
if __name__ == "__main__":
    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)
    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)


    sample_encoder = Encoder(num_layers=2,d_model =512,
                             num_heads=8,
                             dff=2048)
    temp_input = tf.random.uniform((64, 200, 512), dtype=tf.float32, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             maximum_position_encoding=5000)
    temp_target = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_target,
                                  enc_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)


    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)
    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    print(output.shape,attn['decoder_layer2_block2'].shape)
    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)