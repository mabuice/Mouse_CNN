from doctest import REPORTING_FLAGS
import pickle
import os
from .cmouse.mousenet_complete_pool import MouseNetCompletePool
import torch.nn as nn
import pathlib
from .cmouse.anatomy import gen_anatomy
from .mouse_cnn.architecture import Architecture
from .cmouse import network
import pathlib
import os, pdb

def generate_net(retinotopic=False):
    root = pathlib.Path(__file__).parent.resolve()
    cached = os.path.join(root, "data_files", f"net_cache_{'retino' if retinotopic else ''}.pkl")
    if os.path.isfile(cached):
        return network.load_network_from_pickle(cached)
    architecture = Architecture()
    anet = gen_anatomy(architecture)
    net = network.Network(retinotopic=retinotopic)
    net.construct_from_anatomy(anet, architecture)
    network.save_network_to_pickle(net, cached)
    return net

def load(architecture, pretraining=None):
    if architecture not in ("default", "retinotopic"):
        raise ValueError("Architecture must be one of default or retinotopic")  
    
    if pretraining not in (None, "imagenet", "kaiming"):
        raise ValueError("Pretraining must be one of imagenet, kaiming, or None")
    
    #path to this file
    path = pathlib.Path(__file__).parent.resolve()
    # with open(os.path.join(path, "example", "network_complete_updated_number(3,64,64).pkl"), "rb") as file:
    #     net = pickle.load(file)
    #     pdb.set_trace()

    net = generate_net(retinotopic= architecture=="retinotopic")
    mousenet = MouseNetCompletePool(net)
        
    retinomap = None
    if architecture =="retinotopic":
        with open(os.path.join(path, "retinotopics", "retinomap.pkl"), "rb") as file:
            retinomap = pickle.load(file)
    
    model = MouseNetCompletePool(net, retinomap = retinomap)
    

    if pretraining == "kaiming" or None:
        def _kaiming_init_ (m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
        model.apply(_kaiming_init_)

    return model