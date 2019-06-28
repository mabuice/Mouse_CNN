class NetworkLayer:
    def __init__(self, index, sizex, sizey, num_channel, is_input, is_output):
        self.index = index
        self.sizex = sizex
        self.sizey = sizey
        self.num_channel = num_channel
        self.is_input = is_input
        self.is_output = is_output

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
        self.layers = []
        self.connections = []
        
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
        

        

    
