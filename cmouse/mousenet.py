import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import resize_tensor

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MouseNet(nn.Module):
    def __init__(self, network):
        super(MouseNet, self).__init__()
        self.network = network

    def forward(self, x):
        nodes_map = {}
        current_nodes = []
        output_nodes = {}
        for c in self.network.connections:
            if c.pre.is_input == True:
                new_node = nn.Conv2d(
                    c.num_input, c.num_output, c.kernel_size,
                    stride=c.stride)(x)
                new_node = resize_tensor(new_node, c.post.sizex, c.post.sizey)
                nodes_map[c.post.index] = new_node
                current_nodes.append(c.post.index)

        while len(current_nodes) > 0:
            key = current_nodes[0]
            del current_nodes[0]
            node = nodes_map[key]
            if True:
                for c in self.network.connections:
                    if c.pre.index == key:
                        new_node = nn.Conv2d(c.num_input, c.num_output,
                                             c.kernel_size, stride=c.stride)(node)
                        new_node = resize_tensor(new_node, c.post.sizex, c.post.sizey)

                        if c.post.is_output:
                            if c.post.index in output_nodes:
                                # TODO: check channel match
                                output_nodes[c.post.index] = torch.add(
                                output_nodes[c.post.index], new_node)
                            else:
                                output_nodes[c.post.index] = new_node
                        else:
                            if c.post.index in nodes_map:
                                # TODO: check channel match
                                nodes_map[c.post.index] = torch.add(
                                nodes_map[c.post.index], new_node)
                            else:
                                nodes_map[c.post.index] = new_node
                                current_nodes.append(c.post.index)

        out = None
        for key in output_nodes:
            if out is None:
                out = output_nodes[key]
            else:
                # TODO: check channel match
                out = torch.add((out, output_nodes[key]), 3)

        return out
