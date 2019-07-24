from cmouse.gen_network import *
from cmouse.gen_anatomy import *

def test_gen_network():
    anet = gen_anatomy()
    gen_network(anet)
