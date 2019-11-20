import os
import glob
import pandas as pd
import numpy as np
import speechpy
from sklearn.decomposition import PCA, KernelPCA,IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    lengths = list(lengths)

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

def process(data, dimension_reduce = True, derivatives = True, normalize = True):
    """

    :param data:
    :param dimension_reduce:
    :param derivatives:
    :param normalize:
    :return:
    """
    data = np.asarray(data)
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    if dimension_reduce:
        kpca = KernelPCA(n_components=30,kernel='poly')
        data = kpca.fit_transform(data)
    if derivatives:
        feature = data.shape[-1]
        data = speechpy.feature.extract_derivative_feature(data)
        data = data.reshape((data.shape[0], feature * 3))
    return data


def load_data(params):

    cwd = os.path.dirname(os.path.dirname(os.getcwd()))
    mfcc_dir = cwd + "/data/mfcc_{}/trimmed/".format(params.freq)
    eeg_dir = cwd + "/data/eeg_{}/trimmed_feature_{}/".format(params.freq,
                                                              params.feature)
    eeg_paths = glob.glob(eeg_dir+'*.csv')
    eeg_paths.sort()
    mfcc_paths = glob.glob(mfcc_dir + '*.csv')
    mfcc_paths.sort()
    dataset = []
    target = []

    for eeg_path,mfcc_path in zip(eeg_paths,mfcc_paths):
        eeg = pd.read_csv(eeg_path)
        mfcc = pd.read_csv(mfcc_path)
        for i in range(params.number_sentence):
            append_target = mfcc[mfcc['in']==i+1][mfcc.columns[:-1]].values
            append_data = eeg[eeg['in']==i+1][eeg.columns[:-1]].values
            dataset.append(process(append_data))
            target.append(append_target)

    return dataset,target
