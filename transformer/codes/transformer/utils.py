from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA, KernelPCA,IncrementalPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
import numpy as np
import speechpy
import glob
import os
import re


def unpadding(data,length):
    result = []
    for x,y in zip(data,length):
        result.append(x[:int(y)])
    return result


def get_dictionary(label_dir, params):
    with open(label_dir, 'r') as txt:
        data = txt.readlines()
    data = data[:params.number_sentence]
    word_list = []
    raw_text = []
    for line in data:
        sentence = re.findall('\: (.*)\n', line.lower())[0]
        sentence = re.sub(r"[^a-zA-Z]+", ' ', sentence)
        words_vector = []
        sentence = sentence.split(' ')
        for word in sentence:
            if word != '':
                words_vector.append(word.lower())
                word_list.append(word.lower())
        raw_text.append(words_vector)
    word_list = pd.Series(word_list).unique()
    dict = {'<start>': 1, '<end>': 2}
    for word in word_list:
        if word not in dict:
            dict[word] = len(dict) + 1
    return dict

def inverse_attention_format(target):
    result = []
    for i in range(len(target)):
        str_decoded = ''.join([chr(x) for x in np.asarray(target[i]) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        result.append(str_decoded)
    return np.asarray(result)


def convert_inputs_to_attention_format(inputs, target_text, params):
    train_inputs = np.asarray(inputs)
    train_inputs = process(train_inputs)
    train_seq_len = train_inputs.shape[0]
    original = target_text.lower()
    target_text = '<start>'+ target_text.lower() + ' <end>'
    target_text = target_text.lower().split(' ')
    # Transform char into index
    targets = np.asarray([params.dictionary[x] for x in target_text])
    return train_inputs, targets, train_seq_len, original

def load_data(params):
    cwd = os.path.dirname(os.path.dirname(os.getcwd()))
    eeg_dir = cwd + "/data/eeg_{}/trimmed_feature_{}".format(params.freq, params.feature)
    mfcc_dir = cwd + "/data/mfcc_{}/trimmed".format(params.freq)
    label_dir = cwd + "/data/labels_text/sentences.txt"
    eeg_paths = glob.glob(eeg_dir + "/*.csv")
    mfcc_paths = glob.glob(mfcc_dir + "/*.csv")
    input_set = []
    target_set = []
    seq_len_set = []
    original_set = []
    labels_list = list()
    with open(label_dir, 'r') as txt:
        for line in txt.readlines():
            labels_list.append(line.split(':')[-1].split('.')[0])
    labels_list = labels_list[:params.number_sentence]
    dictionary = get_dictionary(label_dir,params)
    params.dictionary = dictionary
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
            append_data, append_targets, append_seq_len, original = convert_inputs_to_attention_format(append_data,
                                                              labels_list[i],params)
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


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def evaluate(inp,params):
    start_token = params.dictionary['<start>']
    end_token = params.dictionary['<end>']

    # inp sentence is portuguese, hence adding the start and end token
    encoder_input = inp

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(params.max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = params.transformer(encoder_input,
                                                     output,
                                                     False,
                                                     None,
                                                     None,
                                                     None)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(inp, label, params, plot= None ):
    result, attention_weights = evaluate(inp, params)
    dictionary = params.dictionary
    dictionary = {v: k for k, v in dictionary.items()}
    predicted_sentence = [dictionary[int(k.numpy())] for k in result[1:]]

    print('Input: {}'.format(label))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, inp, result, plot)