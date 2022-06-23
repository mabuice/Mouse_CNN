import os
from config import OUTPUT_AREAS

PROJECT_NAME = 'cifar10'
DATASET = 'cifar10'
EPOCHS = 1#100#251
LR_EPOCHS = 80
LR = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 64
LOG_INTERVAL = 20
USE_WANDB = 0
MOUSE = 1

RUN_NAME = '%s_LR_%s_M_%s_mousenet'%(PROJECT_NAME, LR, MOMENTUM)

if USE_WANDB:
    WANDB_DRY = 1
 
    if WANDB_DRY:
        os.environ['WANDB_MODE'] = 'dryrun'

WANDB_DIR = os.environ['WANDB_DIR']