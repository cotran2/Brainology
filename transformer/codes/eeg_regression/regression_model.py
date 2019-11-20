import tensorflow as tf
from tensorflow.keras import layers,models
import tcn
import os
import numpy as np
import glob
import pandas as pd
import glob


def regression_model(input_dim = 31,tcn = False):
    if not tcn:
        model = models.Sequential()
        model.add(layers.GRU(128, return_sequences=True, input_shape=(None, input_dim)))
        model.add(layers.TimeDistributed(layers.Dense(13, activation='linear')))
    else:
        model = models.Sequential()
        model.add(TCN(128, return_sequences=True, input_shape=(None, input_dim)))
        model.add(layers.TimeDistributed(layers.Dense(13, activation='linear')))
    model.summary()
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=0.01))
    return model


