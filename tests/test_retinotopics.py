import mousenet
import torch
import pdb

def test_retinotopics_fits_in_memory():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mousenet.load(architecture="retinotopic", pretraining=None)
    model.to(device)
    input = torch.rand(1, 3, 64, 64).to(device)
    results = model(input)
    print(results.shape)
    pdb.set_trace()

def test_stock_fits_in_memory():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mousenet.load(architecture="default", pretraining=None)
    model.to(device)
    input = torch.rand(1, 3, 64, 64).to(device)
    results = model(input)
    print(results.shape)
    pdb.set_trace()