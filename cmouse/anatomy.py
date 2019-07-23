from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
AnatomicalLayerName = namedtuple('AL', ['area', 'depth'])

class AnatomicalLayer:
    def __init__(self, name, num):
        assert(isinstance(name, AnatomicalLayerName))
        self.name = name
        self.num = num
        #self.rf_size = rf_size
        #self.vf_size = vf_size

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

    def draw_graph(self, node_size=1600, node_color='yellow', edge_color='red'):
        G = nx.DiGraph()

        edges = [(p.pre, p.post) for p in self.projections]

        for edge in edges:
            G.add_edge(edge[0], edge[1])
        pos = nx.shell_layout(G)
        node_label_dict = { layer:layer.name.area + layer.name.depth for layer in G.nodes()}

        plt.figure(figsize=(10,10))
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color)
        nx.draw_networkx_labels(G, pos, node_label_dict)
        plt.show()
