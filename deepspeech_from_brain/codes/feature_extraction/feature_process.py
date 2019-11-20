import pandas as pd
import scipy.io.wavfile as wf
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import neurokit as nk
import os
import glob
import scipy.io
import librosa
import numpy as np
import re
from tqdm import tqdm
import sklearn
import gc
from sklearn.decomposition import PCA, KernelPCA,IncrementalPCA
from sklearn.preprocessing import StandardScaler
import utils
from utils import feature_compute, loading_mat

def get_eeg(window_size=10,feature_type = None,output_path= None ,mat_paths= None ):
    """
    get eeg signal
    :param window_size: extracting window size
    :param feature_type: None <-> raw, else [0,1,2]
    :param output_path: output directory
    :param mat_paths: list of mat files
    :return:
    """
    for mat_path in tqdm(mat_paths):
        name = re.findall('mat/(.*)\.mat',mat_path)
        name = name[0].split('_')[0]
        df = loading_mat(mat_path,channels=31)
        if feature_type:
            fc = feature_compute(data=df, window_size=window_size, label=None, map_file=None)
            df = fc.features_computing(trimmed=False,type_tf = ft)
            gc.collect()
            path = output_path + "/feature_{}/".format(feature_type)
        else:
            path = output_path + "/raw/"
        if not os.path.exists(path):
            os.makedirs(path)
        df.to_csv(path + name + '.csv',index=False)


def get_mfcc(signal_rate = 100, output_path = None, audio_paths = None):
    """
    get mfcc
    :param sr: sampling rate
    :param output_path:
    :param audio_paths:
    :return:
    """
    audio = audio_paths[0].split('.')[-1]
    for audio_path in audio_paths:
        name = re.findall('{}\/(.*)\.{}'.format(audio,audio), audio_path)[0]
        mfcc_df = utils.get_mfcc(audio_path,signal_rate)
        #mfcc_df = mfcc_df[(mfcc_df.T == 0).any() == False].reset_index(drop=True)
        path = output_path
        if not os.path.exists(path):
            os.makedirs(path)
        mfcc_df.T.to_csv(path + '/{0}.csv'.format(name), index=False)

def trimm_markers(df = None,time_df = None, ratio = 1 , vad = None):
    """
    trimm the dataframe with corresponding time markers * ratio
    :param df: dataframe
    :param time_df: time markers dataframe
    :param ratio: scale ratio
    :param vad: voice activity detection
    :return:
    """
    df['in'] = 0
    for row in range(len(time_df)):
        start,stop = time_df.iloc[row][['start','stop']]
        start *= ratio
        stop *= ratio
        if vad:
            df.loc[start:stop,'in'] = 1
        else:
            df.loc[start:stop,'in'] = row+1
    if not vad:
        df = df[df['in']!=0].reset_index(drop=True)
    return df


class parameters():

    audio = "wav"
    feature = [1]
    freq_in = 1000
    freq_out = 100
    trim_eeg = True
    trim_mfcc = False
    eeg = True
    mfcc = False
    vad = False
    markers = False
    ratio = 1


if __name__ == "__main__":
    params = parameters
    cwd = os.path.dirname(os.path.dirname(os.getcwd()))
    time_stamps_path = cwd + "/data/time_stamps"
    audio_path = cwd + "/data/" + params.audio
    mat_path = cwd + "/data/mat"

    if params.eeg:
        mat_paths = glob.glob(mat_path+"/*.mat")
        output_path = cwd+"/data/eeg_{}".format(params.freq_out)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ft in params.feature:
            get_eeg(window_size = params.freq_in//params.freq_out,
                    feature_type = ft,
                    output_path = output_path,
                    mat_paths= mat_paths)
    if params.mfcc:
        audio_paths = glob.glob(audio_path + "/*.{}".format(params.audio))
        output_path = cwd + "/data/mfcc_{}".format(params.freq_out)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        get_mfcc(signal_rate=params.freq_out,
                output_path=output_path,
                audio_paths=audio_paths)
    if params.trim_eeg:
        for ft in params.feature:
            eeg_paths = cwd + "/data/eeg_{}/feature_{}".format(params.freq_out,ft)
            time_paths = time_stamps_path
            eeg_paths = glob.glob(eeg_paths+ "/*.csv")
            time_paths = glob.glob(time_paths + "/*.csv")
            eeg_paths.sort()
            time_paths.sort()
            output_path = cwd + "/data/eeg_{}/trimmed_feature_{}/".format(params.freq_out,ft)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for eeg_path, time_path in zip(eeg_paths, time_paths):
                name = eeg_path.split('/')[-1].split('.')[0]
                eeg = pd.read_csv(eeg_path)
                time = pd.read_csv(time_path)
                eeg = trimm_markers(df = eeg,
                                    time_df = time,
                                    ratio = params.ratio,
                                    vad = params.vad)
                eeg.to_csv(output_path + name + '.csv', index=False)
    if params.trim_mfcc:
        mfcc_paths = cwd + "/data/mfcc_{}".format(params.freq_out)
        time_paths = time_stamps_path
        mfcc_paths = glob.glob(mfcc_paths+ "/*.csv")
        time_paths = glob.glob(time_paths + "/*.csv")
        output_path = cwd+ "/data/mfcc_{}/trimmed/".format(params.freq_out)
        mfcc_paths.sort()
        time_paths.sort()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for mfcc_path, time_path in zip(mfcc_paths, time_paths):
            name = mfcc_path.split('/')[-1].split('.')[0]
            mfcc = pd.read_csv(mfcc_path)
            time = pd.read_csv(time_path)
            mfcc = trimm_markers(df = mfcc,
                                 time_df = time,
                                 ratio = params.ratio,
                                 vad = params.vad )
            mfcc.to_csv(output_path + name + '.csv', index=False)




