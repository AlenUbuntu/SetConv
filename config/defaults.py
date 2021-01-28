import os 
from yacs.config import CfgNode as CN 

_C = CN() 

# ------------------------------
# Misc
# ------------------------------
_C.SEED = 0
_C.DEVICE = 'cuda:0'
_C.WEIGHT = ''
_C.OUTPUT_DIR = '.'
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------
_C.DATASET = CN()
_C.DATASET.DIR = '/home/alan/Downloads/imbalanced'
# List of dataset names for training, as listed in path_catalog.py
_C.DATASET.TRAIN = ''
# List of dataset names for valid, as listed in path_catalog.py
_C.DATASET.VALID = ''
# List of dataset names for testing, as listed in path_catalog.py
_C.DATASET.TEST = ''
# ratio to split train and test 
_C.DATASET.RATIO = 0.6
# _C.DATASET.RATIO = 10

# ------------------------------
# DATALOADER
# ------------------------------
_C.DATALOADER = CN()
# Number of data loading threshold
_C.DATALOADER.NUM_WORKERS = 4
# Batch Size
_C.DATALOADER.BATCH_SIZE = 64  # split based on class ratio
# Number of Batches per epoch
_C.DATALOADER.NUM_BATCH = 1000

# ------------------------------
# OPTIMIZATION
# ------------------------------
_C.OPTIMIZER = CN()

_C.OPTIMIZER.NAME = 'adam'
_C.OPTIMIZER.FE_LR = 1e-3  # learning rate for feature extractor
_C.OPTIMIZER.BASE_LR = 1e-2  # learning rate for metric layers

_C.OPTIMIZER.MAX_EPOCH = 20  # maximum epochs
_C.OPTIMIZER.MOMENTUM = 0.9

_C.OPTIMIZER.WEIGHT_DECAY = 5e-4
_C.OPTIMIZER.WEIGHT_DECAY_BIAS = 0

_C.OPTIMIZER.GAMMA = 0.1 
_C.OPTIMIZER.BETAS = (0.9, 0.999)

_C.OPTIMIZER.WARMUP_FACTOR = 1.0 / 3
_C.OPTIMIZER.WARMUP_ITERS = 500
_C.OPTIMIZER.WARMUP_METHOD = 'linear'


# ------------------------------
# MODEL
# ------------------------------
_C.MODEL = CN()
_C.MODEL.INPUT_DIM = 1024
_C.MODEL.DOWNSAMPLE_DIM = (512, )
_C.MODEL.SETCONV_DIM = 256
_C.MODEL.EMBEDDING_DIM = 256
_C.MODEL.MINORITY_CLASS = 1.0

# ------------------------------
# TEST
# ------------------------------
_C.TEST = CN()
_C.TEST.TRAIN_SAMPLE_SIZE = 1000
