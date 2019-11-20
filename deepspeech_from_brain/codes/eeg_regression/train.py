import tensorflow as tf
from tensorflow.keras import layers,models
import os
import numpy as np
import glob
import pandas as pd
from regression_model import *
from utils import *


class parameters():
    freq = 100
    feature = 1
    number_sentence = 30
    batch_size = 100
    tcn = True

if __name__ == '__main__':
    save_path = os.path.dirname(os.path.dirname(os.getcwd()))
    save_path += '/models'
    params = parameters
    data,target = load_data(params)
    if params.batch_size != 1:
        data,_ = pad_sequences(data)
        target,_ = pad_sequences(target)
        print(data.shape)
        input_dim = data.shape[-1]
    else:
        input_dim = data[0].shape[-1]

    regressor = regression_model(input_dim,params.tcn)

    regressor.fit(data,target,batch_size = params.batch_size,
                  validation_split = 0.1, epochs = 100)
    regressor.save_weights(save_path +'/model_{}_{}_{}.hdf5'.format(params.feature,
                                                                 params.freq,
                                                                    params.tcn))
