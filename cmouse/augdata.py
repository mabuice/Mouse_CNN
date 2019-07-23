import sys
sys.path.append('../')
from mouse_cnn.data import *

class AugData(Data):
    def get_kernel_width(self, source_area, source_layer, target_area, target_layer):
        """
        Kernel width associated with an inter-area connection, estimated from voxel model.

        :param source_area: name of source visual area (e.g. 'VISp')
        :param source_layer: name of source layer (e.g. '2/3)
        :param target_area: name of target visual area (e.g. 'VISrl')
        :param target_layer: name of target layer (e.g. '4')
        :return: kernel width (micrometers)
        """
        # TODO: this is here due to a circular dependence between data and voxel (whoops)
        # TODO: consider removing this method; it just wraps target.get_kernel_width

        return 1000
