# Path
DATA_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/'
TRAIN_FEAT_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/train/'
TEST_FEAT_DIR = '/home/soonshin/sss/sr/CORPUS/selvas/test/'
MODEL_SAVED_DIR = '/home/soonshin/sss/sr/selvas-baseline/model_saved/'

# Feature
FILTER_BANK = 64
TRAIN_FEATURE = '.fb64_p'
TEST_FEATURE = '.fb64_p'

# Data loader 
USE_CUDA = True
CUDA_VISIBLE_DEVICES = '2'
USE_SHUFFLE = True
USE_TRANSPOSE = False
VAL_RATIO = 10
BATCH_SIZE = 64
VAL_BATCH_SIZE = 15
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
START = 1
RESUME = False
RESUME_CHECK_POINT = 62
LOG_INTERVAL = 96
N_EPOCHS = 200
ES_PATIENCE = 5
N_CLASSES = 7705
EMBEDDING_SIZE = 1024

# Test
#MODEL_LOAD_DIR = '/home/soonshin/sss/sr/sr-baseline/model_saved/'+'m_190806_090033'
MODEL_LOAD_DIR = '/home/soonshin/sss/sr/selvas-baseline/model_saved'
CHECK_POINT = 27
TEST_FRAMES = 800
THRES = 0.99
PROTOCOL = 'protocol/carspkr01_protocol_1000'
