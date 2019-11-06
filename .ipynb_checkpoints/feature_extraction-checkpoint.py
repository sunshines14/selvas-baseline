import os 
import scipy.io.wavfile as wav
import pickle
import configure as c 
import shutil
import numpy as np
import speechpy
import librosa
from glob import glob
from specAugment import spec_augment_pytorch


def feature_extraction(directory):
    i=1
    #(Car01,CarSpkr01->*_16kHz.wav, Dict02_mix->*_noise.wav, vox_combined->*.wav)
    for wav_file in glob(os.path.join(directory, '**/*.wav'), recursive=True):
        with open(wav_file, 'rb') as f:
            # ===== 1 =====
            (rate, sig) = wav.read(wav_file)
            logfbank = speechpy.feature.lmfe(sig, sampling_frequency=rate, frame_length=0.025, frame_stride=0.01, num_filters=c.FILTER_BANK, fft_length=512, low_frequency=0, high_frequency=None)
            logfbank_cmvn = speechpy.processing.cmvnw(logfbank,win_size=301,variance_normalization=True)
            S = logfbank_cmvn
            
            # ===== 2 =====
            #frame_length = 0.025
            #frame_stride = 0.010 
            #y, sr = librosa.load(wav_file, sr=16000)
            #input_nfft = int(round(sr*frame_length))
            #input_stride = int(round(sr*frame_stride))
            #S = librosa.feature.melspectrogram(y=y, n_mels=128, n_fft=input_nfft, hop_length=input_stride)

            # ===== 3 =====
            #y, sr = librosa.load(wav_file)
            #mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, hop_length=128, fmax=8000)
            #warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)
            #S = warped_masked_spectrogram

            # ===== 4 =====
            #(rate, sig) = wav.read(wav_file)
            #logfbank = speechpy.feature.lmfe(sig, sampling_frequency=rate, frame_length=0.025, frame_stride=0.01, num_filters=c.FILTER_BANK, fft_length=512, low_frequency=0, high_frequency=None)
            #logfbank_cmvn = speechpy.processing.cmvnw(logfbank,win_size=301,variance_normalization=True)
            #warped_masked_logfbank_cmvn = spec_augment_pytorch.spec_augment(mel_spectrogram=logfbank_cmvn)
            #S = warped_masked_logfbank_cmvn

        key = wav_file.split('/')[-3]
        feat_dict = {'label':key, 'feat':S}
        feat_file = wav_file.replace('.wav', '.fb64_p')
        print ([i], key, feat_file)
        with open(feat_file, 'wb') as f:
            pickle.dump(feat_dict, f)
        i=i+1  
            
def test_copy(directory):    
    for p_file in glob(os.path.join(directory, '**/*.p'), recursive=True):
        test_file = p_file.replace('.p', '.t')
        shutil.copy2(p_file, test_file)

if __name__ == '__main__':
    feature_extraction_directory = c.DATA_DIR
    #test_copy_directory = c.TEST_FEAT_DIR
    
    feature_extraction(feature_extraction_directory)
    #test_copy(test_copy_directory)
