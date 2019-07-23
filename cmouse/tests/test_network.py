from cmouse.network import *
from cmouse.mousenet import *
import torch

def test_network():
    net = Network()
    Input = NetworkLayer(0, 100, 100, 3, True, False)
    layer1 = NetworkLayer(1, 100, 100, 4, False, False)
    layer2 = NetworkLayer(2, 100, 100, 12, False, False)
    layer3 = NetworkLayer(3, 100, 100, 4, False, False)
    layer4 = NetworkLayer(4, 100, 100, 2, False, False)
    layer5 = NetworkLayer(5, 100, 100, 6, False, True)

    net.add_layer(Input)
    net.add_layer(layer1)
    net.add_layer(layer2)
    net.add_layer(layer3)
    net.add_layer(layer4)
    net.add_layer(layer5)

    s = 1
    k = 3
    connect1 = Connection(Input, layer1, k, s, 3, 4)
    connect2 = Connection(Input, layer3, k, s, 3, 4)
    connect3 = Connection(Input, layer2, k, s, 3, 12)
    connect4 = Connection(layer1, layer2, k, s, 4, 12)
    connect5 = Connection(layer3, layer2, k, s, 4, 12)
    connect6 = Connection(layer3, layer4, k, s, 4, 2)
    connect7 = Connection(layer2, layer5, k, s, 12, 3)
    connect8 = Connection(layer4, layer5, k, s, 2, 3)

    net.add_connection(connect1)
    net.add_connection(connect2)
    net.add_connection(connect3)
    net.add_connection(connect4)
    net.add_connection(connect5)
    net.add_connection(connect6)
    net.add_connection(connect7)
    net.add_connection(connect8)

    mnet = MouseNet(net)
    a = torch.randn(2, 3, 100, 100)
    mnet(a)
    return net
