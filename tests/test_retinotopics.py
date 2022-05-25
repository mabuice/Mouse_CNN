import mousenet
import torch

def test_fits_in_memory():
    model = mousenet.load(architecture="retinotopic", pretraining=None)
    input = 
