import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch import nn
from config import *

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = int(2 * self.gsw * EDGE_Z + 1)

class ConvLayer:
    def __init__(self, source_name, target_name, params, out_size, out_sigma):
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size
        self.out_sigma = out_sigma


class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """
    def __init__(self):
        self.layers = []
        
    def find_conv_source_target(self, source_name, target_name):
        for layer in self.layers:
            if layer.source_name == source_name and layer.target_name == target_name:
                return layer
        assert('no conv layer found!')
    
    def find_conv_target_area(self, target_name):
        for layer in self.layers:
            if layer.target_name == target_name:
                return layer
        assert('no conv layer found!')
        
    def construct_from_anatomy(self, anet):
        # construct conv layer for input -> LGNv
        LGNv_out = np.floor(anet.find_layer('LGNv','').num/INPUT_SIZE[0]/INPUT_SIZE[1]/INPUT_SIZE[2])
        out_size =  INPUT_SIZE[1]*anet.find_layer('LGNv','').sigma
        out_sigma = anet.find_layer('LGNv','').sigma
       
        convlayer = ConvLayer('input', 'LGNv', 
                              ConvParam(in_channels=INPUT_SIZE[0], 
                                        out_channels=LGNv_out,
                                        gsh=INPUT_GSH,
                                        gsw=INPUT_GSW),
                              out_size, out_sigma)
        self.layers.append(convlayer)
       
        # construct conv layers for all other connections
        G, _ = anet.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        for i, e in enumerate(nx.edge_bfs(G, root)):
            in_layer_name = e[0].area+e[0].depth
            out_layer_name = e[1].area+e[1].depth
            
            in_conv_layer = self.find_conv_target_area(in_layer_name)
            in_size = in_conv_layer.out_size
            in_channels = in_conv_layer.params.out_channels
            
            out_anat_layer = anet.find_layer(e[1].area, e[1].depth)
            
            out_size = in_size * out_anat_layer.sigma
            out_sigma = out_anat_layer.sigma
            out_channels = np.floor(out_anat_layer.num/out_size**2)
            
            proj = anet.find_projection(e[0].area, e[0].depth, e[1].area, e[1].depth)
            
            convlayer = ConvLayer(in_layer_name, out_layer_name, 
                                  ConvParam(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            gsh=proj.gsh,
                                            gsw=proj.gsw), 
                                  out_size, out_sigma)
            self.layers.append(convlayer)
            
    def make_graph(self):
        G = nx.DiGraph()
        edges = [(p.source_name, p.target_name) for p in self.layers]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer:layer for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=1600, node_color='yellow', edge_color='red'):
        G, node_label_dict = self.make_graph()
        edge_label_dict = {(c.source_name, c.target_name):(c.params.in_channels, c.params.out_channels) for c in self.layers}
        plt.figure(figsize=(14,20))
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,alpha=0.5)
        nx.draw_networkx_labels(G, pos, node_label_dict)
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict)
        plt.show()   




class Conv2dMask(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, stride=1, padding=0):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, 
                                         stride=stride, padding=padding)
        self.mask = torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw))
        
    def forward(self, input):
        return super(Conv2dMask, self).conv2d_forward(input, self.weight*self.mask)

    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)

        x = np.arange(-width, width+1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)
        return np.random.rand(len(x), len(x)) < probability




class MouseNet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network):
        super(MouseNet, self).__init__()
        self.Convs = nn.ModuleDict()
        self.network = network
        
        G, _ = network.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        self.edge_bfs = [e for e in nx.edge_bfs(G, root)] # traversal edges by bfs
        
        for e in self.edge_bfs:
            layer = network.find_conv_source_target(e[0], e[1])
            params = layer.params
            padding = int((params.kernel_size-1/layer.out_sigma)/2)
           
            self.Convs[e[0]+e[1]] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                               params.gsh, params.gsw, stride=int(1/layer.out_sigma), padding=padding)

        final_layer = network.find_conv_source_target('VISpor2/3','VISpor5') 
        final_size = final_layer.out_size
        final_channels = final_layer.params.out_channels
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * final_size * final_size, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )

    def forward(self, x):
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                calc_graph[e[1]] = self.Convs[e[0]+e[1]](x)
            else:
                if e[1] in calc_graph:
                    calc_graph[e[1]] = calc_graph[e[1]] + self.Convs[e[0]+e[1]](calc_graph[e[0]])
                else:
                    calc_graph[e[1]] = self.Convs[e[0]+e[1]](calc_graph[e[0]])

        x = torch.flatten(calc_graph['VISpor5'], 1)
        x = self.classifier(x)
        return x
                
