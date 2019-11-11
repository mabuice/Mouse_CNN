from cmouse.network import *
from cmouse.anatomy import *
import torch
import sys
sys.path.append('../')

def test_network():
    anet = gen_anatomy(data_folder='./data_files')
    net = Network()
    net.construct_from_anatomy(anet)
    x = torch.randn(10, 2,50,50)
    return MouseNet(net)(x)
    
