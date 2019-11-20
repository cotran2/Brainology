import tensorflow as tf
from keras import layers, optimizers, models
import tcn
from tcn import TCN
from keras import backend as K
from CTCModel import CTCModel

def ctc_model(input_dim,nb_labels,padding_value,regression = True):

    reg = regressor(input_dim)
    i = models.Input(batch_shape=(None, None, input_dim))
    o = layers.Masking(mask_value=padding_value)(i)
    if regression:
        o = reg(o)
    o = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.1))(o)
    o = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.1))(o)
    o = layers.Bidirectional(layers.GRU(32, return_sequences=True, dropout=0.1))(o)
    o = layers.TimeDistributed(layers.Dense(nb_labels, activation='softmax'))(o)
    model = CTCModel([i], [o])
    model.compile(optimizer=optimizers.Adam(lr=1e-2))
    return model
def regressor(input_dim):

    model_path = '/home/gautam-admin/EEG/deepspeech_from_brain/models/model_1_100_True.hdf5'
    i = models.Input(batch_shape=(None, None, input_dim))

    o = TCN(return_sequences=True)(i)  # The TCN layers are here.
    o = layers.TimeDistributed(layers.Dense(13, activation='linear'))(o)

    regressor = models.Model(inputs=[i], outputs=[o])
    regressor.load_weights(model_path)

    return regressor