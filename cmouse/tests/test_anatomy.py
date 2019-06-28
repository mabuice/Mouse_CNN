from cmouse.anatomy import *

def test_Projection():
    pre = AnatomicalLayer(AnatomicalLayerName('VISp', 'L4'), 10000, 10, 10)
    post1 = AnatomicalLayer(AnatomicalLayerName('VISl', 'L4'), 20000, 5, 5)
    post2 = AnatomicalLayer(AnatomicalLayerName('VISp', 'L23'), 10000, 10, 10)
    proj1 = AreaProjection(pre, post1, 0.1, 2)
    proj2 = LaminarProjection(pre, post2, 0.2, 1)

def test_AnatomicalNet():
    l1 = AnatomicalLayer(AnatomicalLayerName('VISp', 'L4'), 10000, 10, 10)
    l2 = AnatomicalLayer(AnatomicalLayerName('VISl', 'L4'), 20000, 5, 5)
    l3 = AnatomicalLayer(AnatomicalLayerName('VISp', 'L23'), 10000, 10, 10)
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
