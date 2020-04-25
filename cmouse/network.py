import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch import nn
from config import *
import os
import pickle
from anatomy import gen_anatomy
import matplotlib.pyplot as plt

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, out_sigma):
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
        
    def construct_from_anatomy(self, anet, data):
        # construct conv layer for input -> LGNv
        self.area_channels['input'] = INPUT_SIZE[0]
        self.area_size['input'] = INPUT_SIZE[1]
        
        out_sigma = anet.find_layer('LGNv','').sigma
        out_channels = np.floor(anet.find_layer('LGNv','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2])
        data.set_num_channels('LGNv', '', out_channels)
        self.area_channels['LGNv'] = out_channels
        
        out_size =  INPUT_SIZE[1] * out_sigma
        self.area_size['LGNv'] = out_size

       
        convlayer = ConvLayer('input', 'LGNv', 
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
            
           
            out_sigma = out_anat_layer.sigma
            out_size = in_size * out_sigma
            self.area_size[e[1].area+e[1].depth] = out_size
            out_channels = np.floor(out_anat_layer.num/out_size**2)
            
            data.set_num_channels(e[1].area, e[1].depth, out_channels)
            self.area_channels[e[1].area+e[1].depth] = out_channels
            
            convlayer = ConvLayer(in_layer_name, out_layer_name, 
                                  ConvParam(in_channels=in_channels, 
                                            out_channels=out_channels,
                                        gsh=data.get_kernel_peak_probability(e[0].area, e[0].depth, e[1].area, e[1].depth),
                                        gsw=data.get_kernel_width_pixels(e[0].area, e[0].depth, e[1].area, e[1].depth), out_sigma=out_sigma),
                                    out_size)
            
            self.layers.append(convlayer)
            
    def make_graph(self):
        G = nx.DiGraph()
        edges = [(p.source_name, p.target_name) for p in self.layers]
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        #node_label_dict = { layer:(layer, c, int(self.area_channels[layer])) for layer in G.nodes()}
        node_label_dict = { layer:'%s\n%s'%(layer, int(self.area_channels[layer])) for layer in G.nodes()}
        
        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color='yellow', edge_color='red'):
        G, node_label_dict = self.make_graph()
        edge_label_dict = {(c.source_name, c.target_name):(c.params.kernel_size) for c in self.layers}

        plt.figure(figsize=(12,12))
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,alpha=0.4)
        nx.draw_networkx_labels(G, pos, node_label_dict, fontsize=1000,font_weight=640, alpha=0.7, font_color='black')
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict, fontsize=2000, font_weight=640,alpha=0.7, font_color='red')
        plt.show()  


class Conv2dMask(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=1, stride=1, padding=0):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if mask:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        else:
            self.mask = None
    def forward(self, input):
        if self.mask is not None:
            return super(Conv2dMask, self).conv2d_forward(self.mypadding(input), self.weight*self.mask)
        else:
            return super(Conv2dMask, self).conv2d_forward(self.mypadding(input), self.weight)
            
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

        re = np.random.rand(len(x), len(x)) < probability
        plt.imshow(re, cmap='Greys')
        return re



class MouseNet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=1, bn=1):
        super(MouseNet, self).__init__()
        self.Convs = nn.ModuleDict()
        self.bn = bn
        if self.bn:
            self.BNs = nn.ModuleDict()
        self.network = network
        
        G, _ = network.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph
        self.edge_bfs = [e for e in nx.edge_bfs(G, root)] # traversal edges by bfs
        
        for e in self.edge_bfs:
            layer = network.find_conv_source_target(e[0], e[1])
            params = layer.params

            #print('_______________________________')
            #print(e[0], e[1])
#            print(params.in_channels, params.out_channels, params.kernel_size, params.gsh, params.gsw, params.stride, params.padding)
      

            self.Convs[e[0]+e[1]] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                               params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
            
            ## plotting Gaussian 
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))

            if self.bn:
                self.BNs[e[0]+e[1]] = nn.BatchNorm2d(params.out_channels)

        total_size=0
        for area in OUTPUT_AREAS:
            if area =='VISp5':
                layer = network.find_conv_source_target('VISp2/3','VISp5')
                visp_out = layer.params.out_channels
                total_size += 32*32*32
            else:
                layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
                total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        if 'VISp5' in OUTPUT_AREAS:
            self.visp5_downsampler = nn.Conv2d(visp_out, 32, 1, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(int(total_size), HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        )

    def get_img_feature(self, x, area_list):
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x))
                else:
                    calc_graph[e[1]] = self.Convs[e[0]+e[1]](x)
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + self.Convs[e[0]+e[1]](calc_graph[e[0]])
                else:
                    if self.bn:
                        calc_graph[e[1]] = self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                    else:
                        calc_graph[e[1]] = self.Convs[e[0]+e[1]](calc_graph[e[0]])
            calc_graph[e[1]] = nn.ReLU(inplace=True)(calc_graph[e[1]])
        
        if len(area_list) == 1:
            return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                else:
                    re=torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
        return re


    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS)

        x = self.classifier(x)
        return x
                
    def _initialize_weights(self):
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.BatchNorm2d):
                   nn.init.constant_(m.weight, 1)
                   nn.init.constant_(m.bias, 0)
               elif isinstance(m, nn.Linear):
                   nn.init.normal_(m.weight, 0, 0.01)
                   nn.init.constant_(m.bias, 0)

def gen_network(net_name, architecture, data_folder=DATA_DIR):
    if os.path.exists('./myresults/%s.pkl'%net_name):
        f = open('./myresults/%s.pkl'%net_name,'rb')
        net = pickle.load(f)
    else:
        anet = gen_anatomy(architecture)
        net = Network()
        net.construct_from_anatomy(anet, architecture)
        f = open('./myresults/%s.pkl'%net_name,'wb')
        pickle.dump(net, f)
    return net