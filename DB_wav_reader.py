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

def find_train_feats_SA(directory, pattern='**/*'+c.TRAIN_FEATURE_SA):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def find_test_feats(directory, pattern='**/*'+c.TEST_FEATURE):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def read_DB_structure(directory):
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

def read_train_feats_structure_SA(directory):
    DB = pd.DataFrame()
    DB['filename'] = find_train_feats(directory) 
    DB['filename'] = DB['filename'].apply(lambda x: x.replace('\\', '/'))
    DB['speaker_id'] = DB['filename'].apply(lambda x: x.split('/')[-3])
    DB['dataset_id'] = DB['filename'].apply(lambda x: x.split('/')[-6])
    num_speakers = len(DB['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB.head(10))

    DB_SA = pd.DataFrame()
    DB_SA['filename'] = find_train_feats_SA(directory) 
    DB_SA['filename'] = DB_SA['filename'].apply(lambda x: x.replace('\\', '/'))
    DB_SA['speaker_id'] = DB_SA['filename'].apply(lambda x: x.split('/')[-3])
    DB_SA['dataset_id'] = DB_SA['filename'].apply(lambda x: x.split('/')[-6])
    num_speakers = len(DB_SA['speaker_id'].unique())
    logging.info('Found {} files with {} different speakers.'.format(str(len(DB_SA)).zfill(7), str(num_speakers).zfill(5)))
    logging.info(DB_SA.head(10))

    tmp = [DB, DB_SA]
    DB_merge = pd.concat(tmp)
    return DB_merge

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
    train_DB = read_train_feats_structure_SA(train_dir)
    test_DB = read_test_feats_structure(test_dir)
    return train_DB, test_DB

if __name__ == '__main__':
    train_DB, test_DB = test()
    print (train_DB)
    print (test_DB)