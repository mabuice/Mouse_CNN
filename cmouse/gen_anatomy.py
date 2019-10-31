import sys
sys.path.append('../')
from mouse_cnn.data import *
from anatomy import *

class AugData(Data):
    def get_kernel_width(self, source_area, source_layer, target_area, target_layer):
        return 1000

def gen_anatomy(input_depths = ['4'],
                output_depths = ['2/3', '5'],
                lamimar_connections = [('4', '2/3'), ('2/3', '5')],
                data_folder = '../data_files/'):
    anet = AnatomicalNet()
    data = AugData(data_folder=data_folder)
    areas = data.get_areas()
    depths = data.get_layers()
    output_map = {} # collect output layers for each hierarchy

    # create LGNv
    hierarchy = 0
    layer_name = AnatomicalLayerName('LGNv', '')
    layer0 = AnatomicalLayer(layer_name, data.get_num_neurons('LGNv', None))
    anet.add_layer(layer0)
    output_map[0] = [layer0]

    for hierarchy in [1,2,3]:
        output_map[hierarchy] = []
        for area in areas:
            if data.get_hierarchical_level(area) == hierarchy:
                # create anatomical module for one area
                # add layers
                area_layers = {}
                for depth in depths:
                    layer_name = AnatomicalLayerName(area, depth)
                    layer = AnatomicalLayer(layer_name, data.get_num_neurons(area, depth))
                    area_layers[depth] = layer
                    anet.add_layer(layer)
                    if depth in output_depths:
                        output_map[hierarchy].append(layer)

                # add LaminarProjection
                for source, target in lamimar_connections:
                    p = data.get_hit_rate_peak(source, target)
                    w = data.get_hit_rate_width(source, target)
                    anet.add_projection(LaminarProjection(area_layers[source], area_layers[target], p, w))

                # add AreaProjection
                for depth in depths:
                    if depth in input_depths:
                        for l in output_map[hierarchy-1]:
                            e = data.get_extrinsic_in_degree(area, depth)
                            w = data.get_kernel_width(l.name.area, l.name.depth, area, depth)
                            anet.add_projection(AreaProjection(l, area_layers[depth], e, w))
    return anet
