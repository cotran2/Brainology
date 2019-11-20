import pandas as pd
import scipy.io.wavfile as wf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from scipy import signal
import scipy.io.wavfile as wav
import neurokit as nk
import os
import glob
import scipy.io
import librosa
import numpy as np
import re
import pywt
import sklearn
import gc
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Input, Dense
from keras.models import Model
from numpy.fft import fft
from numpy import zeros, floor, log10, log, mean, array, sqrt, vstack, cumsum, ones, log2, std
from numpy.linalg import svd, lstsq


def loading_mat(file_path, channels):
    """
    Purpose : loading and preprocess .mat EEG files
    Arg:
        file_path : file directory
        channels : number of eeg channels processed in .mat file
    Output:
        df : dataframe for the EEG signals
    """
    mat = scipy.io.loadmat(file_path)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    names = list(mat.keys())
    df = pd.DataFrame()
    for key in names[:channels]:
        value = mat[key]
        value = pd.DataFrame(value)
        df = pd.merge(df, value, how='right', right_index=True, left_index=True)
    df.columns = names[:channels]
    return df


def get_mfcc(file_path, sr_):
    """
    Purpose : compute mfcc features from a audio file
    Arg:
        file_path : audio file
    Output:
        mf : dataframe for mfcc features
    """
    #     (rate,sig) = wav.read(file_path)
    #     mfcc_feat = mfcc(sig,rate,winstep=0.01)
    y, sr = librosa.load(file_path, sr=sr_)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr_, hop_length=1, n_mfcc=13)
    mf = pd.DataFrame(mfcc_feat)
    return mf


def get_audio_signal(file_path, sr):
    """
    Purpose : compute audio signal from a audio file
    Arg:
        file_path : audio file
    Output:
        y : dataframe for audio signal features
    """
    y, sr = librosa.load(file_path, sr=sr)
    y = pd.DataFrame(y)
    return y


def trimm_mfcc(file_path, label):
    """
    Trimming the mfcc by the label mapping
    Arg:
        file_path : the mfcc path
        label : label_mapping.csv
    Output:
        mf_df : trimmed mfcc file
    """
    mf_df = get_mfcc(file_path)
    pre_len = 0
    mf_df['code'] = None
    mf_df['word'] = None
    for row in range(len(label)):
        dif, code, word = label.iloc[row][['dif', 'code', 'word']]
        dif = round(dif, 1)
        if dif == 0:
            dif = 0.1
        if pre_len == 0:
            start = int(pre_len) * 100
            end = int(pre_len * 100) + int(dif * 100) - 1
        else:
            start = pre_len
            end = pre_len + int(dif * 100)
        mf_df.at[start:end, ['code', 'word']] = code, word
        pre_len = end
    mf_df = mf_df[mf_df['code'].notnull()]
    return mf_df


def hurst(X):
    """
    Compute the Hurst exponent of X.
    """
    N = len(X)
    T = array([float(i) for i in range(1, N + 1)])
    Y = cumsum(X)
    Ave_T = Y / T

    S_T = zeros((N))
    R_T = zeros((N))
    for i in range(N):
        S_T[i] = std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = lstsq(n[1:], R_S[1:])[0]
    return H[0]


def first_order_diff(X):
    """ Compute the first order difference of a time series.

        For a time series X = [x(1), x(2), ... , x(N)], its	first order
        difference is:
        Y = [x(2) - x(1) , x(3) - x(2), ..., x(N) - x(N-1)]

    """
    D = []

    for i in range(1, len(X)):
        D.append(X[i] - X[i - 1])

    return D


def pfd(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    """
    if D is None:
        D = first_order_diff(X)
    N_delta = 0;  # number of sign changes in derivative of the signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            N_delta += 1
    n = len(X)
    return log10(n) / (log10(n) + log10(n / n + 0.4 * N_delta))


def hfd(X, Kmax):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter
    """
    L = [];
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(log(mean(Lk)))
        x.append([log(float(1) / k), 1])

    (p, r1, r2, s) = lstsq(x, L)
    return p[0]


class feature_compute():
    def __init__(self, data, window_size, label, map_file):
        self.data = data
        self.window_size = window_size
        self.label = label
        self.map_file = map_file

    def hurst(self, col_data):
        """
        Purpose : compute hurst accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data.tolist()
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            df.append(hurst(data[i * window_size:(i + 1) * window_size]))
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def pfd(self, col_data):
        """
        Purpose : compute pfd accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data.tolist()
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            df.append(pfd(data[i * window_size:(i + 1) * window_size]))
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def hfd(self, col_data):
        """
        Purpose : compute pfd accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data.tolist()
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            df.append(hfd(data[i * window_size:(i + 1) * window_size], 100))
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def band_spectral_entropy(self, col_data):
        """
        Purpose : compute spectral entropy accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data
        band = [0.5, 4, 7, 12, 30]
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            complexity = nk.complexity(data[i * window_size:(i + 1) * window_size], spectral=True, shannon=False,
                                       sampen=False, multiscale=False, svd=False, correlation=False,
                                       higushi=False, petrosian=False, fisher=False, hurst=False, dfa=False,
                                       lyap_r=False, lyap_e=False, bands=band)
            df.append(complexity['Entropy_Spectral'])
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def spectral_entropy(self, col_data):
        """
        Purpose : compute spectral entropy accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            complexity = nk.complexity(data[i * window_size:(i + 1) * window_size], spectral=True, shannon=False,
                                       sampen=False, multiscale=False, svd=False, correlation=False,
                                       higushi=False, petrosian=False, fisher=False, hurst=False, dfa=False,
                                       lyap_r=False, lyap_e=False)
            df.append(complexity['Entropy_Spectral'])
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def kurtosis(self, col_data):
        """
        Purpose : compute kurtosis accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data
        window_size = self.window_size
        df = []
        for i in range(int(len(data) / window_size)):
            df.append(data[i * window_size:(i + 1) * window_size].kurtosis())
        df = pd.DataFrame(df)
        return df.reset_index(drop=True)

    def average(self, col_data):
        """
        Purpose : compute moving average accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data
        window_size = self.window_size
        df = pd.DataFrame()
        for i in range(int(len(data) / window_size)):
            dt = np.array(data[i * window_size:(i + 1) * window_size])
            window = np.ones((window_size,)) / float(window_size)
            dt = pd.DataFrame(np.convolve(dt, window, mode='valid'))
            df = pd.concat([df, dt])
        return df.reset_index(drop=True)

    def short_time_fourier_transform(self, col_data):
        """
        Purpose : compute short time fourier transform
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing short time fourier transform
        Arg:
        """
        data = col_data
        window_size = self.window_size
        if window_size == 2:
            f, t, Zxx = signal.stft(data, 100, nperseg=4)
        else:
            f, t, Zxx = signal.stft(data, 100, nperseg=2)
        Zxx = np.abs(Zxx)
        Zxx = np.max(Zxx, axis=0)
        print(round(len(Zxx) / len(data), 2))
        Zxx = Zxx[:(len(data) // window_size)]
        df = pd.DataFrame(Zxx)
        return df

    def wavelet_transform(self, col_data):
        data = col_data
        window_size = self.window_size
        ca_df = []
        cd_df = []
        for i in range(int(len(data) / window_size)):
            (cA, cD) = pywt.dwt(data[i * window_size:(i + 1) * window_size], 'db4')
            ca_ = nk.complexity(cA, spectral=True, shannon=False, sampen=False, multiscale=False, svd=False,
                                correlation=False,
                                higushi=False, petrosian=False, fisher=False, hurst=False, dfa=False, lyap_r=False,
                                lyap_e=False)
            cd_ = nk.complexity(cD, spectral=True, shannon=False, sampen=False, multiscale=False, svd=False,
                                correlation=False,
                                higushi=False, petrosian=False, fisher=False, hurst=False, dfa=False, lyap_r=False,
                                lyap_e=False)
            ca_df.append(ca_['Entropy_Spectral'])
            cd_df.append(cd_['Entropy_Spectral'])
        df = pd.DataFrame({'CA': ca_df, 'CD': cd_df})
        return df.reset_index(drop=True)

    def root_mean_square(self, col_data):
        """
        Purpose : compute root mean square accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data
        window_size = self.window_size
        df = pd.DataFrame()
        for i in range(int(len(data) / window_size)):
            dt = np.power(data[i * window_size:(i + 1) * window_size], 2)
            window = np.ones((window_size,)) / float(window_size)
            dt = pd.DataFrame(np.sqrt(np.convolve(dt, window, mode='valid')))
            df = pd.concat([df, dt])
        return df.reset_index(drop=True)

    def zero_crossing_rate(self, col_data):
        """
        Purpose : compute root mean square accross the data
        Arg:
            data : input EEG signals data
            window_sze : moving window size
        Output:
            df : results after computing moving spectral entropy
        """
        data = col_data.values
        window_size = self.window_size
        df = pd.DataFrame()
        df = librosa.feature.zero_crossing_rate(data, frame_length=100, hop_length=10)
        df = pd.DataFrame(np.transpose(df))
        return df.reset_index(drop=True)

    def features_computing(self, trimmed=True, type_tf=0):
        """
        Purpose : take the EEG signals data and computing moving features
        Arg:
            self:
        Output:
            df: the dataframe with 5 statistical features for each channels
        """
        if trimmed == True:
            data = self.trimmed_data(rate=1000)
            labeling = data[['code', 'word']]
            data = data.drop(columns=['code', 'word'])
        else:
            data = self.data
        window_size = self.window_size
        df = pd.DataFrame()
        df_step = pd.DataFrame()
        if type_tf == 1:
            for column in data.columns:
                col_data = data[column]
                zero = self.zero_crossing_rate(col_data)
                rms = self.root_mean_square(col_data)
                avg = self.average(col_data)
                kurt = self.kurtosis(col_data)
                entropy = self.spectral_entropy(col_data)
                df_step = pd.concat([zero, rms, avg, kurt, entropy], axis=1)
                df_step.columns = [column + '_' + feature for feature in ['zero', 'rms', 'avg', 'kurt', 'entropy']]
                df = pd.concat([df, df_step], axis=1)
        elif type_tf == 2:
            for column in data.columns:
                col_data = data[column]
                sfft = self.short_time_fourier_transform(col_data)
                # if len(sfft) <= 0.2 * len(col_data):
                #     print('something wrong')
                wavelet = self.wavelet_transform(col_data)
                df_step = pd.concat([sfft, wavelet['CA'], wavelet['CD']], axis=1)
                df_step.columns = [column + '_' + feature for feature in ['sfft', 'cA', 'cD']]
                df = pd.concat([df, df_step], axis=1)
        elif type_tf == 3:
            for column in data.columns:
                col_data = data[column]
                hur = self.hurst(col_data)
                pfd = self.pfd(col_data)
                #                 hfd = self.hfd(col_data)
                bse = self.band_spectral_entropy(col_data)
                df_step = pd.concat([hur, pfd, bse], axis=1)
                df_step.columns = [column + '_' + feature for feature in ['hurst', 'pdf', 'spectral_entropy']]
                df = pd.concat([df, df_step], axis=1)
        return df

    def mapping_label(self):
        """
        Purpose : mapping the words with unique codes
        Arg:
            self.label : label dataframe
        Output:
            label : mapped label dataframe
        """
        label = self.label
        map_file = self.map_file
        label = label.merge(map_file, how='left', on='word')
        label['dif'] = label['end_time'] - label['start_time']
        return label

    def trimmed_data(self, rate):
        """
        Purpose : trimmed the data with label start times and end times
        Arg:
            self.label : label file
        Output:
            df : trimmed EEG signals with label
        """
        data = self.data
        label = self.mapping_label()
        data['code'] = None
        data['word'] = None
        df = pd.DataFrame(columns=data.columns)
        for row in range(len(label)):
            start_row = label.iloc[row]['start_time'] * rate
            end_row = label.iloc[row]['end_time'] * rate
            if start_row == end_row:
                end_row = end_row + rate / 10
            code = label.iloc[row]['code']
            word = label.iloc[row]['word']
            df_append = data.iloc[int(start_row):int(end_row)]
            df_append['code'] = code
            df_append['word'] = word
            df = df.append(df_append)
        return df

    def label_data(self):
        """
        Purpose : re-label the data
        Arg:
        Output:
            data : trimmed EEG signals with label
        """
        label = self.mapping_label()
        data = self.features_computing()
        pre_len = 0
        data['code'] = None
        data['word'] = None
        for row in range(len(label)):
            dif, code, word = label.iloc[row][['dif', 'code', 'word']]
            dif = round(dif, 1)
            if dif == 0:
                dif = 0.1
            if pre_len == 0:
                start = int(pre_len) * 100
                end = int(pre_len * 100) + int(dif * 100) - 1
            else:
                start = pre_len
                end = pre_len + int(dif * 100)
            data.at[start:end, ['code', 'word']] = code, word
            pre_len = end
        return data


def encode(df, encoding_dim):
    """
        Encoding 
    """
    X = df.values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    ncol = X.shape[1]

    input_dim = Input(shape=(ncol,))
    # DEFINE THE DIMENSION OF ENCODER ASSUMED 3
    encoding_dim = encoding_dim
    # DEFINE THE ENCODER LAYERS

    encoded1 = Dense(155, activation='relu')(input_dim)
    encoded2 = Dense(100, activation='relu')(encoded1)
    encoded3 = Dense(55, activation='relu')(encoded2)
    encoded4 = Dense(30, activation='relu')(encoded3)
    encoded5 = Dense(15, activation='relu')(encoded4)
    encoded6 = Dense(encoding_dim, activation='relu')(encoded5)
    # DEFINE THE DECODER LAYERS

    decoded1 = Dense(15, activation='relu')(encoded5)
    decoded2 = Dense(30, activation='relu')(decoded1)
    decoded3 = Dense(55, activation='relu')(decoded2)
    decoded4 = Dense(100, activation='relu')(decoded3)
    decoded5 = Dense(155, activation='sigmoid')(decoded4)
    decoded6 = Dense(ncol, activation='relu')(encoded5)
    # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
    autoencoder = Model(input=input_dim, output=decoded5)
    # CONFIGURE AND TRAIN THE AUTOENCODER
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()
    autoencoder.fit(X, X, nb_epoch=500, batch_size=100, shuffle=True, validation_data=(X, X))
    # THE ENCODER TO EXTRACT THE REDUCED DIMENSION FROM THE ABOVE AUTOENCODER
    encoder = Model(input=input_dim, output=encoded5)
    encoded_input = Input(shape=(encoding_dim,))
    encoded_out = encoder.predict(X)
    encoded_out = pd.DataFrame(encoded_out)
    #     encoded_out = pd.concat([encoded_out,y],axis=1)
    return encoded_out


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths