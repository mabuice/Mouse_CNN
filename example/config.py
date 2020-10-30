import os
import argparse
import os 

INPUT_SIZE=(3,64,64)
NUM_CLASSES = 1000
HIDDEN_LINEAR = 2048

EDGE_Z = 1 #Z-score (# standard deviations) of edge of kernel
INPUT_GSH = 1 #Gaussian height of input to LGNv 
INPUT_GSW = 4 #Gaussian width of input to LGNv

#OUTPUT_AREAS = ['VISpor5']
OUTPUT_AREAS = ['VISp5', 'VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5']


def get_out_sigma(source_area, source_depth, target_area, target_depth):
    if target_depth == '4':
        if target_area != 'VISp' and target_area != 'VISpor':
            return 1/2
        if target_area == 'VISpor':
            if source_area == 'VISp':
                return 1/2    
    return 1


