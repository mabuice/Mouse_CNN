import torch
from torch import nn
import networkx as nx
import numpy as np
from config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES

class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if mask == 0:
            self.mask = None
        if mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel(gsh, gsw, kernel_size, out_channels, in_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")

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
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re

class MouseNet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, bn=1):
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

            self.Convs[e[0]+e[1]] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                               params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if self.bn:
                self.BNs[e[0]+e[1]] = nn.BatchNorm2d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        for area in OUTPUT_AREAS:
            if area =='VISp5':
                layer = network.find_conv_source_target('VISp2/3','VISp5')
                visp_out = layer.params.out_channels
                # create 1x1 Conv downsampler for VISp5
                visp_downsample_channels = 32
                ds_stride = 2
                self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
                total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
            else:
                layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
                total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
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
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the flattened activation of that area; 
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}
        for e in self.edge_bfs:
            if e[0] == 'input':
                if self.bn:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](x)))
                else:
                    calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](x))
            else:
                if e[1] in calc_graph:
                    if self.bn:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = calc_graph[e[1]] + nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
                else:
                    if self.bn:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.BNs[e[0]+e[1]](self.Convs[e[0]+e[1]](calc_graph[e[0]])))
                    else:
                        calc_graph[e[1]] = nn.ReLU(inplace=True)(self.Convs[e[0]+e[1]](calc_graph[e[0]]))
        
        if len(area_list) == 1:
            return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
        else:
            re = None
            for area in area_list:
                if area == 'VISp5':
                    re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                else:
                    if re is not None:
                        re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                    else:
                        re = torch.flatten(calc_graph[area], 1)
        return re

    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)