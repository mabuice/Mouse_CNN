import numpy as np
import networkx as nx
from anatomy import gen_anatomy
import torch
from torch import nn
from example.config import INPUT_SIZE, EDGE_Z, INPUT_GSH, INPUT_GSW, SUBFIELDS, get_out_sigma, get_resolution
import os
import pickle
import matplotlib.pyplot as plt

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, out_sigma):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param gsh: Gaussian height for generating Gaussian mask 
        :param gsw: Gaussian width for generating Gaussian mask
        :param out_sigma: ratio between output size and input size, 1/2 means reduce output size to 1/2 of the input size
        """

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = 2*int(self.gsw * EDGE_Z) + 1

        KmS = int((self.kernel_size-1/out_sigma))
        if np.mod(KmS,2)==0:
            padding = int(KmS/2)
        else:
            padding = (int(KmS/2), int(KmS/2+1), int(KmS/2), int(KmS/2+1))
        self.padding = padding
        self.stride = int(1/out_sigma)
        
class ConvLayer:
    def __init__(self, source_name, target_name, params, out_size):
        """
        :param params: ConvParam containing the parameters of the layer
        :param source_name: name of the source area, e.g. VISp4, VISp2/3, VISp5
        :param target_name: name of the target area
        :param out_size: output size of the layer
        """
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size


class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """
    def __init__(self):
        self.layers = []
        self.area_channels = {}
        self.area_size = {}
        
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
        
    def construct_from_anatomy(self, anet, architecture):
        """
        construct network from anatomy 
        :param anet: anatomy class which contains anatomical connections
        :param architecture: architecture class which calls set_num_channels for calculating connection strength
        """
        # construct conv layer for input -> LGNd
        self.area_channels['input'] = INPUT_SIZE[0]
        self.area_size['input'] = INPUT_SIZE[1]
        
        out_sigma = 1
        out_channels = np.floor(anet.find_layer('LGNd','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2])
        architecture.set_num_channels('LGNd', '', out_channels)
        self.area_channels['LGNd'] = out_channels
        
        out_size =  INPUT_SIZE[1] * out_sigma
        self.area_size['LGNd'] = out_size
       
        convlayer = ConvLayer('input', 'LGNd',
                              ConvParam(in_channels=INPUT_SIZE[0], 
                                        out_channels=out_channels,
                                        gsh=INPUT_GSH,
                                        gsw=INPUT_GSW, out_sigma=out_sigma),
                              out_size)
        self.layers.append(convlayer)
       
        # construct conv layers for all other connections
        G, _ = anet.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        for i, e in enumerate(nx.edge_bfs(G, root)):
            
            in_layer_name = e[0].area+e[0].depth
            out_layer_name = e[1].area+e[1].depth
            print('constructing layer %s: %s to %s'%(i, in_layer_name, out_layer_name))
            
            in_conv_layer = self.find_conv_target_area(in_layer_name)
            in_size = in_conv_layer.out_size
            in_channels = in_conv_layer.params.out_channels
            
            out_anat_layer = anet.find_layer(e[1].area, e[1].depth)
            
            out_sigma = get_out_sigma(e[0].area, e[0].depth, e[1].area, e[1].depth)
            out_size = in_size * out_sigma
            self.area_size[e[1].area+e[1].depth] = out_size

            if SUBFIELDS:
                pixel_area = calculate_pixel_area_with_visual_field(architecture, e[1].area, e[1].depth)
                out_channels = np.floor(out_anat_layer.num / pixel_area)
            else:
                out_channels = np.floor(out_anat_layer.num/out_size**2)

            architecture.set_num_channels(e[1].area, e[1].depth, out_channels)
            self.area_channels[e[1].area+e[1].depth] = out_channels
            
            convlayer = ConvLayer(in_layer_name, out_layer_name, 
                                  ConvParam(in_channels=in_channels, 
                                            out_channels=out_channels,
                                        gsh=architecture.get_kernel_peak_probability(e[0].area, e[0].depth, e[1].area, e[1].depth),
                                        gsw=architecture.get_kernel_width_pixels(e[0].area, e[0].depth, e[1].area, e[1].depth), out_sigma=out_sigma),
                                    out_size)
            
            self.layers.append(convlayer)
            
    def make_graph(self):
        """
        produce networkx graph
        """
        G = nx.DiGraph()
        edges = [(p.source_name, p.target_name) for p in self.layers]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer:'%s\n%s'%(layer, int(self.area_channels[layer])) for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color='yellow', edge_color='red'):
        """
        draw the network structure
        """
        G, node_label_dict = self.make_graph()
        edge_label_dict = {(c.source_name, c.target_name):(c.params.kernel_size) for c in self.layers}
        plt.figure(figsize=(12,12))
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,alpha=0.4)
        nx.draw_networkx_labels(G, pos, node_label_dict, font_size=10,font_weight=640, alpha=0.7, font_color='black')
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict, font_size=20, font_weight=640,alpha=0.7, font_color='red')
        plt.show()  


def calculate_pixel_area_with_visual_field(architecture, area, depth):
    out_field = architecture.get_visual_field(area)
    out_width_degrees = out_field[1] - out_field[0]
    out_height_degrees = out_field[3] - out_field[2]
    out_pixels_per_degree = get_resolution(area, depth)
    out_area_pixels_squared = out_width_degrees * out_height_degrees * out_pixels_per_degree ** 2
    return out_area_pixels_squared


def gen_network_from_anatomy(architecture):
    anet = gen_anatomy(architecture)
    net = Network()
    net.construct_from_anatomy(anet, architecture)
    return net

def save_network_to_pickle(net, file_path):
    f = open(file_path,'wb')
    pickle.dump(net, f)

def load_network_from_pickle(file_path):
    f = open(file_path,'rb')
    net = pickle.load(f)
    return net

def gen_network(net_name, architecture, data_folder=''):
    file_path = './myresults/%s.pkl'%net_name
    if os.path.exists(file_path):
        net = load_network_from_pickle(file_path)
    else:
        net = gen_network_from_anatomy(architecture)
        if not os.path.exists('./myresults'):
            os.mkdir('./myresults')
        save_network_to_pickle(net, file_path)
    return net
