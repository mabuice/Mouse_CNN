import mousenet
import torch

def test_retinotopics_fits_in_memory():
    retinotopic_model = mousenet.load(architecture="retinotopic", pretraining=None)
    