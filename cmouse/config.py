INPUT_SIZE=(2, 64, 64)
NUM_CLASSES = 10
HIDDEN_LINEAR = 4096
DATA_FOLDER = '/home/iris/Dropbox/iris-shared-workspace/Mouse_CNN/data/'
WANDB_DIR = '/home/iris/Dropbox/iris-shared-workspace/Mouse_CNN/wandb/'

EDGE_Z = 1 #Z-score (# standard deviations) of edge of kernel
INPUT_GSH = 1 #Gaussian height of input to LGNv 
INPUT_GSW = 4 #Gaussian width of input to LGNv

OUTPUT_AREA = 'VISpor'

def get_output_shrinkage(area, depth):
    if depth == '4':
        if area!='LGNv' and area!='VISp' and area!='VISpor':
            return 1/2
    return 1
