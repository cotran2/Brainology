from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA, KernelPCA,IncrementalPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import speechpy
import glob
import os

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def unpadding(data,length):
    result = []
    for x,y in zip(data,length):
        result.append(x[:int(y)])
    return result

def inverse_ctc_format(target):
    result = []
    for i in range(len(target)):
        str_decoded = ''.join([chr(x) for x in np.asarray(target[i]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        result.append(str_decoded)
    return np.asarray(result)


def convert_inputs_to_ctc_format(inputs, target_text):
    train_inputs = np.asarray(inputs)
    train_inputs = process(train_inputs)
    train_seq_len = train_inputs.shape[0]
    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '')
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                          for x in targets])
    return train_inputs, targets, train_seq_len, original

def load_data(params):

    cwd = os.path.dirname(os.path.dirname(os.getcwd()))
    eeg_dir = cwd + "/data/eeg_{}/trimmed_feature_{}".format(params.freq, params.feature)
    mfcc_dir = cwd + "/data/mfcc_{}/trimmed".format(params.freq)
    label_dir = cwd + "/labels_text/sentences.txt"
    eeg_paths = glob.glob(eeg_dir + "/*.csv")
    mfcc_paths = glob.glob(mfcc_dir + "/*.csv")

    labels_list = list()
    with open(label_dir, 'r') as txt:
        for line in txt.readlines():
            labels_list.append(line.split(':')[-1].split('.')[0])
    labels_list = labels_list[:params.number_sentence]
    input_set = []
    target_set = []
    seq_len_set = []
    original_set = []

    for mfcc_path, eeg_path in zip(mfcc_paths, eeg_paths):
        if params.data == 'fusion':
            mfcc = pd.read_csv(mfcc_path)
            mfcc = mfcc.rename(columns={'in': 'sentence'})
            eeg = pd.read_csv(eeg_path)
            mfcc = mfcc[mfcc.columns[:-1]]
            df = pd.concat([mfcc, eeg], axis=1)
        elif params.data == 'mfcc':
            mfcc = pd.read_csv(mfcc_path)
            mfcc = mfcc.rename(columns={'in': 'sentence'})
            df = mfcc
        elif params.data == 'eeg':
            eeg = pd.read_csv(eeg_path)
            df = eeg
        else:
            raise ('Invalide type of data input')

        for i in range(params.number_sentence):
            df.columns = [str(i) for i in range(len(df.columns) - 1)] + ['sentence']
            append_data = df[df['sentence'] == i + 1][df.columns[:-1]].values
            append_data, append_targets, append_seq_len, original = convert_inputs_to_ctc_format(append_data,
                                                              labels_list[i])
            input_set.append(append_data)
            seq_len_set.append(append_seq_len)
            target_set.append(append_targets)
            original_set.append(original)
    return input_set, target_set, seq_len_set, original_set

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


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


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
