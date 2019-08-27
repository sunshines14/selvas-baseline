# Path
DATA_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/train/'
TRAIN_FEAT_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/train/'
TEST_FEAT_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/test/'
MODEL_SAVED_DIR = '/home/soonshin/sss/sr/selvas-baseline/model_saved/'

# Feature
FILTER_BANK = 64
TRAIN_FEATURE = '.fb64_p'
TEST_FEATURE = '.fb64_p'
#FILTER_BANK = 128
#TRAIN_FEATURE = '.melspec128_p'
#TEST_FEATURE = '.melspec128_p'
#FILTER_BANK = 256
#TRAIN_FEATURE = '.specAugment_p'
#TEST_FEATURE = '.specAugment_p'

# Data loader 
USE_CUDA = True
CUDA_VISIBLE_DEVICES = '1'
USE_SHUFFLE = True
USE_TRANSPOSE = False
VAL_RATIO = 10
BATCH_SIZE = 64
VAL_BATCH_SIZE = 20
NUM_WIN_SIZE = 300

# Optimizer
OPTIMIZER = 'sgd'
LR = 1e-2
WD = 1e-4

# Leaning rate scheduler
FACTOR = 0.1
LR_PATIENCE = 2
MIN_LR = 1e-4 

# Trainer
START = 34
RESUME = True
RESUME_CHECK_POINT = 33
LOG_INTERVAL = 96
N_EPOCHS = 100
ES_PATIENCE = 5
N_CLASSES = 7705
EMBEDDING_SIZE = 512

# Test
#MODEL_LOAD_DIR = '/home/soonshin/sss/sr/sr-baseline/model_saved/'+'m_190806_090033'
MODEL_LOAD_DIR = '/home/soonshin/sss/sr/selvas-baseline/model_saved'
CHECK_POINT = 36
TEST_FRAMES = 800
THRES = 0.99
PROTOCOL = 'protocol/carspkr01_protocol_100'
