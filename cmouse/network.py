import networkx as nx
import matplotlib.pyplot as plt

class NetworkLayer:
    def __init__(self, index, sizex, sizey, num_channel):
        self.index = index
        self.sizex = sizex
        self.sizey = sizey
        self.num_channel = num_channel

class Connection:
    def __init__(self, pre, post, kernel_size, stride,
                 num_input, num_output):
        assert(isinstance(pre, NetworkLayer))
        assert(isinstance(post, NetworkLayer))
        self.pre = pre
        self.post = post
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_input = num_input
        self.num_output = num_output

class Network:
    def __init__(self):
        self.input_layer = None
        self.layers = []
        self.connections = []

    def add_input_layer(self, layer):
        assert(self.input_layer == None)
        self.add_layer(layer)
        self.input_layer = layer

    def add_layer(self, layer):
        assert(isinstance(layer, NetworkLayer))
        for l in self.layers:
            if l.index == layer.index:
                print("Layer already exist!")
                return
        self.layers.append(layer)

    def add_connection(self, connection):
        assert(isinstance(connection, Connection))
        for c in self.connections:
            if c.pre == connection.pre and c.post == connection.post:
                print("Connection already exist!")
                return
        self.connections.append(connection)

    def find_layer(self, layer_index):
        assert(isinstance(layer_index, int))
        for layer in self.layers:
            if layer.index == layer_index:
                return layer
        print("No layer with index %d found!"%layer_index)

    def is_terminal_layer(self, layer):
        assert(isinstance(layer, NetworkLayer))
        for c in self.connections:
            if c.pre == layer:
                return False
        return True

    def draw_graph(self, node_size=1600, node_color='yellow', edge_color='red'):
        G=nx.DiGraph()

        edges = [(c.pre.index, c.post.index) for c in self.connections]
        edge_label_dict = {(c.pre.index, c.post.index):(c.kernel_size, c.stride, c.num_input, c.num_output) for c in self.connections}
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        pos = nx.shell_layout(G)
        node_label_dict = { layer_index: (layer_index, self.find_layer(layer_index).num_channel) for layer_index in G.nodes()}

        nx.draw(G, pos, with_labels=False, node_size=node_size, node_color=node_color, edge_color=edge_color)
        nx.draw_networkx_labels(G, pos, node_label_dict)
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict)
        plt.show()
