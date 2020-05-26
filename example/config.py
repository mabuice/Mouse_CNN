import os
import argparse
import os 

DATA_DIR = os.environ['DATA_DIR']
RESULT_DIR = os.environ['RESULT_DIR']
INPUT_SIZE=(3,64,64)
NUM_CLASSES = 10
HIDDEN_LINEAR = 2048

EDGE_Z = 1 #Z-score (# standard deviations) of edge of kernel
INPUT_GSH = 1 #Gaussian height of input to LGNv 
INPUT_GSW = 4 #Gaussian width of input to LGNv

#OUTPUT_AREAS = ['VISpor5']
OUTPUT_AREAS = ['VISp5', 'VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5']


def get_output_shrinkage(area, depth):
    if depth == '4':
        if area!='LGNv' and area!='VISp' and area!='VISpor':
            return 1/2
    return 1
