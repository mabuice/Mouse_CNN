from collections import namedtuple
import networkx as nx
AnatomicalLayerName = namedtuple('AL', ['area', 'depth'])

class AnatomicalLayer:
    def __init__(self, name, num, rf_size, vf_size):
        assert(isinstance(name, AnatomicalLayerName))
        self.name = name
        self.num = num
        self.rf_size = rf_size
        self.vf_size = vf_size

class Projection:
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post

class LaminarProjection(Projection):
    def __init__(self, pre, post, vp, vw):
        assert(pre.name.area == post.name.area)
        Projection.__init__(self, pre, post)
        self.vp = vp
        self.vw = vw

class AreaProjection(Projection):
    def __init__(self, pre, post, dp, dw):
        assert(pre.name.area != post.name.area)
        Projection.__init__(self, pre, post)
        self.dp = dp
        self.dw = dw

class AnatomicalNet:
    def __init__(self):
        self.layers = []
        self.projections = []

    def find_layer(self, layer):
        for l in self.layers:
            if l.name == layer.name:
                return True
        return False

    def find_projection(self, projection):
        for proj in self.projections:
            if ( proj.pre.name == projection.pre.name and
                 proj.post.name == projection.post.name ):
                return True
        return False

    def add_layer(self, layer):
        assert(isinstance(layer, AnatomicalLayer))
        if self.find_layer(layer):
            print("%s already exist!"%(layer.name))
            return
        self.layers.append(layer)

    def add_projection(self, projection):
        assert(isinstance(projection, Projection))
        if self.find_projection(projection):
            print("Projection %s to %s already exist!"%(projection.pre.name,
                                                        projection.post.name))
            return
        self.projections.append(projection)

    def make_graph(self):
        graph = nx.DiGraph()
        for layer in self.layers:
            graph.add_node(layer.name)
        for proj in self.projections:
            graph.add_edge(proj.pre.name, proj.post.name)
        return graph
