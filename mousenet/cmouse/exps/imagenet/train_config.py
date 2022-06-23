import os
from config import OUTPUT_AREAS

PROJECT_NAME = 'imagenet'

#vgg16
#EPOCHS = 100#110#251
#LR_EPOCHS = 40
#BATCH_SIZE = 256
#NET = 92
#SCALE = 0.1 #transform scale

EPOCHS = 1#70#110#251
LR_EPOCHS = 50#40#200
LR_EPOCHS2 = 10
BATCH_SIZE = 256
NET = 1
SCALE = 0.4 #transform scale
LR = 0.01
MOMENTUM = 0.9
LOG_INTERVAL = 20

USE_WANDB = 0
NUM_WORKERS = 10

RUN_NAME = '%s_Net_%s_LR_%s_MO_%s'%(PROJECT_NAME, NET, LR, MOMENTUM)

if USE_WANDB:
    WANDB_DRY = 0
 
    if WANDB_DRY:
        os.environ['WANDB_MODE'] = 'dryrun'

WANDB_DIR = os.environ['WANDB_DIR']
IMAGENET_DIR = os.environ['IMAGENET_DIR']