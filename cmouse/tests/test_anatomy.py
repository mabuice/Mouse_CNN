from cmouse.anatomy import *

def test_Projection():
    pre = AnatomicalLayer('VISp', 'L4', 10000)
    post1 = AnatomicalLayer('VISl', 'L4', 2000)
    post2 = AnatomicalLayer('VISp', 'L23', 10000)
    proj1 = AreaProjection(pre, post1, 0.1, 2)
    proj2 = LaminarProjection(pre, post2, 0.2, 1)

def test_AnatomicalNet():
    l1 = AnatomicalLayer('VISp', 'L4', 10000)
    l2 = AnatomicalLayer('VISl', 'L4', 20000)
    l3 = AnatomicalLayer('VISp', 'L23', 10000)
    proj1 = AreaProjection(l1, l2, 0.1, 2)
    proj2 = LaminarProjection(l1, l3, 0.2, 1)
    proj3 = AreaProjection(l3, l2, 0.1, 1)

    ANet = AnatomicalNet()
    ANet.add_layer(l1)
    ANet.add_layer(l2)
    ANet.add_layer(l3)
    ANet.add_projection(proj1)
    ANet.add_projection(proj2)
    ANet.add_projection(proj3)
    ANet.make_graph()
