import logging
import os
import librosa
import numpy as np
import pandas as pd
import configure as c 
from glob import glob


#np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def find_wavs(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_train_feats(directory, pattern='**/*'+c.TRAIN_FEATURE):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_test_feats(directory, pattern='**/*'+c.TEST_FEATURE):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_train_DB_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_wavs(directory) 
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) 
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) 
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6]) 
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def read_test_DB_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_wavs(directory) 
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/')) 
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3]) 
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6]) 
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def read_train_feats_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_train_feats(directory) 
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3])
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6])
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB

def read_test_feats_structure(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_test_feats(directory)
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3])
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6])
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))
    return DB
 
def test():
    train_dir = c.TRAIN_FEAT_DIR
    test_dir = c.TEST_FEAT_DIR
    #train_data_DB = read_train_DB_structure(train_dir)
    #test_data_DB = read_test_DB_structure(test_dir)
    train_DB = read_train_feats_structure(train_dir)
    test_DB = read_test_feats_structure(test_dir)
    return train_DB, test_DB
    #return train_data_DB, test_data_DB

if __name__ == '__main__':
    train_DB, test_DB = test()
    print (train_DB)
    print (test_DB)
    #train_data_DB, test_data_DB = test()
    #print (train_data_DB)
    #print (test_data_DB)
