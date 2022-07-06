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

SUBFIELDS = False # use area-specific visual subfields


def get_out_sigma(source_area, source_depth, target_area, target_depth):
    source_resolution = get_resolution(source_area, source_depth)
    target_resolution = get_resolution(target_area, target_depth)
    return target_resolution / source_resolution


def get_resolution(area, depth):
    """
    :param area: cortical visual area name
    :param depth: layer name
    :return: model resolution in pixels per degree visual angle
    """
    if area == 'VISp' or area == 'LGNd':
        return 1
    else:
        return 0.5
