import numpy as np
from mouse_cnn.data import Data
from mouse_cnn.voxel import Target, get_surface_area_mm2

#TODO: review hard-coded values for LGN

class Architecture(Data):
    """
    Extends Data with kernel parameter calculations. To get kernel parameters for
    an inter-area connection, it is necessary to first call set_num_channels for
    the source area/layer. This information is needed for example to convert kernel
    width estimates in micrometers to kernel width estimates in pixels.
    """

    def __init__(self):
        self.targets = _get_targets(self)
        self.channels = {}

    def set_num_channels(self, area, layer, channels):
        """
        :param area: visual area name
        :param layer: layer name
        :param channels: number of channels in convolutional model of this area/layer
        """
        self.channels[_get_name(area, layer)] = channels

    def _get_pixels_per_micrometer(self, area, layer):
        name = _get_name(area, layer)

        if not name in self.channels.keys():
            raise Exception('Please first call set_num_channels for {} layer {}'.format(area, layer))

        channels = self.channels[name]
        n = self.get_num_neurons(area, layer)
        linear_pixels = np.sqrt(n / channels)
        mm2 = get_surface_area_mm2(name)
        linear_micrometers = np.sqrt(mm2) * 1000

        return linear_pixels / linear_micrometers

    def get_kernel_peak_probability(self, source_area, source_layer, target_area, target_layer):
        if source_area == target_area: # from interlaminar hit rates
            return self.get_hit_rate_peak(source_layer, target_layer)
        elif 'LGN' in source_area:
            return 1
        else: # from mesoscale model
            target = self.targets[_get_name(target_area, target_layer)]
            source_name = _get_name(source_area, source_layer)
            e_ij = target.get_n_external_inputs_for_source(source_name)
            d_w = self.get_kernel_width_pixels(source_area, source_layer, target_area, target_layer)
            source_channels = self.channels[source_name]
            d_p = e_ij / (source_channels * 2 * np.pi * d_w ** 2)
            return d_p

    def get_kernel_width_pixels(self, source_area, source_layer, target_area, target_layer):
        if source_area == target_area: # from interlaminar hit rate spatial profile
            width_micrometers = self.get_hit_rate_width(source_layer, target_layer)
            return width_micrometers * self.get_pixels_per_micrometer(source_area, source_layer)
        elif 'LGN' in source_area:
            return 3
        else: # from mesoscale model
            target = self.targets[_get_name(target_area, target_layer)]
            width_mm = target.get_kernel_width_mm(_get_name(source_area, source_layer))
            return width_mm * 1000 * self.get_pixels_per_micrometer(source_area, source_layer)


def _get_name(area, layer):
    return area + layer


def _get_targets(data):
    # build dictionary of voxel.Target instances per target layer
    targets = {}
    for area in data.get_areas():
        if data.get_hierarchical_level(area) > 0:
            for layer in data.get_layers():
                in_degree = data.get_extrinsic_in_degree(area, layer)
                targets[_get_name(area, layer)] = Target(area, layer, external_in_degree=in_degree)
    return targets


if __name__ == '__main__':
    pass
