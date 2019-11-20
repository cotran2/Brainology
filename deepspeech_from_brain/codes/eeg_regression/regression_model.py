import os
import numpy as np
import glob
import pandas as pd


def regression_model(input_dim = 31,tcn = False):
    if not tcn:
        from tensorflow.keras import layers, models
        import tensorflow as tf

        model = models.Sequential()
        model.add(layers.GRU(128, return_sequences=True, input_shape=(None, input_dim)))
        model.add(layers.TimeDistributed(layers.Dense(13, activation='linear')))
        model.summary()
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=0.01))
    else:
        import tcn
        import keras
        from keras.layers import Dense,TimeDistributed
        from keras.models import Input, Model
        from tcn import TCN
        i = Input(batch_shape=(None, None, input_dim))

        o = TCN(return_sequences=True)(i)  # The TCN layers are here.
        o = TimeDistributed(Dense(13, activation='linear'))(o)

        model = Model(inputs=[i], outputs=[o])
        model.summary()
        model.compile(loss='mse',
                      optimizer= keras.optimizers.Adam(lr=0.01))

    return model


