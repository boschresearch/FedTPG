"""
Modified from https://github.com/KaiyangZhou/Dassl.pytorch

Copyright (c) 2020 Kaiyang, licensed under the MIT License
cf. 3rd-party-licenses.txt file in the root directory of this source tree

"""

from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1
# _C.NUM_CLIENT = 1
# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True

_C.EXP_NAME = ""
###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ""
_C.DATASET.NAME_SPACE = []
# Number of images per class
_C.DATASET.NUM_SHOTS = 4
_C.DATASET.TESTNAME_SPACE = []
###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Setting for the train_x data-loader
_C.DATALOADER.TRAIN = CN()
_C.DATALOADER.TRAIN.BATCH_SIZE = 128
_C.DATALOADER.TRAIN.CLASSES = None
# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.BATCH_SIZE = 128
_C.DATALOADER.TEST.CLASSES = None

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ""
_C.MODEL.N_CTX = 4 # number of context vectors
_C.MODEL.D_CTX = 1 # number of layers with context vectors
_C.MODEL.CTX_INIT = ""  # initialization words
_C.MODEL.NAME = 'fedtpg'
_C.MODEL.DEPTH = 0 # number of self-attention modules
###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "sgd"
_C.OPTIM.LR = 0.003
_C.OPTIM.WEIGHT_DECAY = 1e-5
_C.OPTIM.MOMENTUM = 0.9

# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "cosine"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 1000


###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to only save the last one
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
############## knoledege guidance
_C.TRAIN.W = 8.0
_C.TRAIN.NUM_CLASS_PER_CLIENT =10
_C.TRAIN.AVAIL_PERCENT = 1
_C.TRAIN.SPLIT = 'base'
###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# Use test or val set for FINAL evaluation
_C.TEST.SPLIT = "new"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"
# _C.TEST.FINAL_MODEL = "best_val"






