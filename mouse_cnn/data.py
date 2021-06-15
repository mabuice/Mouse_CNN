# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
from scipy.optimize import curve_fit

"""
Interface to mouse data sources.
"""


class Data:
    def __init__(self, data_folder = 'data_files'):
        # self.e18 = Ero2018(data_folder)
        self.p11 = Perin11()
        self.b19 = Billeh19()

    def get_areas(self):
        """
        :return: list of names of visual areas included in the model
        """
        # return ['LGNd', 'VISp', 'VISal', 'VISpor']

        #return ['LGNd', 'VISp', 'VISl', 'VISpor']

        return ['LGNd', 'VISp', 'VISl', 'VISrl', 'VISli', 'VISpl', 'VISal', 'VISpor']
    def get_layers(self):
        """
        :return: list of cortical layers included in model
        """
        return ['2/3', '4', '5']

    def get_hierarchical_level(self, area):
        """
        :param area: Name of visual area
        :return: Hierarchical level number, from 0 (LGN) to 3 (VISpor) from Stefan's
            analysis
        """
        hierarchy = {
            'LGNd': 0,
            'VISp': 1,
            'VISl': 2, 'VISrl': 2, 'VISli': 2, 'VISpl': 2, 'VISal': 2,
            'VISpor': 3
        }
        return hierarchy[area]

    def get_num_neurons(self, area, layer):
        """
        :param area: visual area name (e.g. 'VISp')
        :param layer: layer name (e.g. '2/3')
        :return: estimate of number of excitatory neurons in given area/layer
        """
        numbers = { 'LGNd':21200,
                    'VISp2/3': 173253,
                    'VISl2/3': 22299,
                    'VISrl2/3': 22598,
                    'VISli2/3': 9587,
                    'VISpl2/3': 17924,
                    'VISal2/3': 15760,
                    'VISpor2/3': 30576,
                    'VISp4': 108623,
                    'VISl4': 15501,
                    'VISrl4': 14360,
                    'VISli4': 5620,
                    'VISpl4': 3912,
                    'VISal4': 9705,
                    'VISpor4': 5952,
                    'VISp5': 134530,
                    'VISl5': 20826,
                    'VISrl5': 19173,
                    'VISli5': 11611,
                    'VISpl5': 20041,
                    'VISal5': 15939,
                    'VISpor5': 30230}
        if area == 'LGNd':
            region = area
        else:
            region = '%s%s'%(area, layer) 
        return numbers[region]

    # def get_num_neurons(self, area, layer):
    #     """
    #     :param area: visual area name (e.g. 'VISp')
    #     :param layer: layer name (e.g. '2/3')
    #     :return: estimate of number of excitatory neurons in given area/layer
    #     """
    #     #TODO: compare with other estimates

    #     if area in ['VISrl', 'VISli', 'VISpor']:
    #         # Ero et al. doesn't include these estimates, so we approximate from
    #         # density of VISl. Specifically we multiply the estimate from VISl by
    #         # the ratio of surface areas of L2/3. Surface areas estimated from
    #         # convex hull of flat map of voxels.

    #         surface_areas_23 = {
    #             'VISl': 0.9279064282165419,
    #             'VISrl': 0.698045549856564,
    #             'VISli': 0.43560916751267514,
    #             'VISpor': 1.3936554078054724
    #         }

    #         ratio = surface_areas_23[area] / surface_areas_23['VISl']
    #         return self.e18.get_n_excitatory('VISl', layer) * ratio

    #     return self.e18.get_n_excitatory(area, layer)

    def get_extrinsic_in_degree(self, target_area, target_layer):
        """
        :param target_area: visual area name (e.g. 'VISp')
        :param target_layer: layer name (e.g. '2/3')
        :return: estimate of mean number of neurons from OTHER AREAS that synapse onto a
            single excitatory neuron in given area / layer
        """
        return 1000 #TODO: replace with real estimate

    def get_hit_rate_peak(self, source_layer, target_layer):
        """
        :param source_layer: name of presynaptic layer
        :param target_layer: name of postsynaptic layer
        :return: fraction of excitatory neuron pairs with functional connection in this
            direction, at zero horizontal offset
        """
        hit_rate = self.b19.get_connection_probability(source_layer, target_layer)
        fraction_of_peak = np.exp(-75**2 / 2 / self.get_hit_rate_width(source_layer, target_layer)**2)
        return hit_rate / fraction_of_peak

    def get_hit_rate_width(self, source_layer, target_layer):
        """
        :param source_layer: name of presynaptic layer
        :param target_layer: name of postsynaptic layer
        :return: width of Gaussian approximation of fraction of excitatory neuron pairs with
            functional connection in this direction
        """

        # Levy & Reyes (2012; J Neurosci) report sigma 114 micrometers for probability of
        # functional connection between pairs of L4 pyramidal cells in mouse auditory cortex.
        # See their Table 3.
        l4_to_l4 = 114

        # Stepanyants, Hirsch, Martinez (2008; Cerebral Cortex) report variations in width
        # of potential connection probability depending on source and target layer in cat V1.
        # See their Figure 8B. Values below are rough manual estimates from their figure.
        cat = { # source -> target
            '2/3': {'2/3': 225, '4': 50, '5': 100, '6': 50},
            '4': {'2/3': 220, '4': 180, '5': 140, '6': 110},
            '5': {'2/3': 150, '4': 100, '5': 210, '6': 125},
            '6': {'2/3': 120, '4': 20, '5': 150, '6': 150}
        }

        return cat[source_layer][target_layer] / cat['4']['4'] * l4_to_l4

    def get_visual_field_shape(self, area):
        """
        :param area: visual area name
        :return: (height, width) of visual field for that area
        """
        # We return a constant for simplicity. This is based on the range of the scale
        # bars in Figure 9C,D of ﻿J. Zhuang et al., “An extended retinotopic map of mouse cortex,”
        # Elife, p. e18372, 2017. In fact different areas have different visual field shapes and
        # offsets, but we defer this aspect to future models.
        return (55, 90)


# class Ero2018:
#     """
#     Data from supplementary material of:

#     Erö, C., Gewaltig, M. O., Keller, D., & Markram, H. (2019). A Cell Atlas for the Mouse Brain.
#     Frontiers in Neuroinformatics, 13, 7.
#     """

#     def __init__(self, data_folder):
#         file_name = data_folder + '/Data_Sheet_1_A Cell Atlas for the Mouse Brain.CSV'

#         if not os.path.isfile(file_name):
#             raise Exception('Missing data file {}, available from {}'.format(
#                 file_name,
#                 'https://www.frontiersin.org/articles/10.3389/fninf.2018.00084/full#supplementary-material'
#             ))

#         self.regions = []
#         self.excitatory = []
#         with open(file_name) as csvfile:
#             r = csv.reader(csvfile)
#             header_line = True
#             for row in r:
#                 if header_line:
#                     header_line = False
#                 else:
#                     self.regions.append(row[0])
#                     self.excitatory.append(row[4])

#     def get_n_excitatory(self, area, layer=None):
#         area_map = {
#             'LGNd': 'Dorsal part of the lateral geniculate complex',
#             'LGNv': 'Ventral part of the lateral geniculate complex',
#             'VISal': 'Anterolateral visual area',
#             'VISam': 'Anteromedial visual area',
#             'VISl': 'Lateral visual area',
#             'VISp': 'Primary visual area',
#             'VISpl': 'Posterolateral visual area',
#             'VISpm': 'posteromedial visual area'
#         }

#         if layer is None:
#             index = self.regions.index(area_map[area])
#             result = np.int(self.excitatory[index])
#         elif layer == '6':
#             index_a = self.regions.index('{} layer 6a'.format(area_map[area]))
#             index_b = self.regions.index('{} layer 6a'.format(area_map[area]))
#             result = np.int(self.excitatory[index_a]) + np.int(self.excitatory[index_b])
#         else:
#             index = self.regions.index('{} layer {}'.format(area_map[area], layer))
#             result = np.int(self.excitatory[index])

#         return result


class Perin11:
    """
    This class fits a Gaussian function to the connection probability vs. inter-somatic
    distance among pairs of thick-tufted L5 pyramids in P14-16 Wistar rats, from Fig. 1 of [1].

    In the source figure, I would expect "overall" to be the sum of reciprical
    and non-reciprocal, but it isn't. It doesn't look like this much affects the spatial
    profile though, just the peak (which we don't use).

    [1] R. Perin, T. K. Berger, and H. Markram, “A synaptic organizing principle for cortical neuronal
    groups.,” Proc. Natl. Acad. Sci. U. S. A., vol. 108, no. 13, pp. 5419–24, Mar. 2011.
    """

    def __init__(self):
        connection_probability_vs_distance = [
                [17.441860465116307, 0.21723833429098494],
                [52.79069767441864, 0.1676015362748359],
                [87.44186046511628, 0.14761544742492516],
                [122.5581395348837, 0.12294674448846282],
                [157.67441860465118, 0.09515710527111632],
                [192.55813953488376, 0.10208848701121961],
                [227.44186046511635, 0.06337617564339071],
                [262.5581395348837, 0.03480630235582299],
                [297.44186046511635, 0.07021622765899538]]

        def gaussian(x, peak, sigma):
            return peak * np.exp(-x ** 2 / 2 / sigma ** 2)

        cp = np.array(connection_probability_vs_distance)
        popt, pcov = curve_fit(gaussian, cp[:,0], cp[:,1], p0=(.2, 150))
        self.width_micrometers = popt[1]


class Billeh19():
    """
    Data from literature review by Yazan Billeh.
    TODO: further details and reference the paper once it's published.
    """

    def __init__(self):
        self._layers = ['2/3', '4', '5', '6']
        self.probabilities = [
            [.160, .016, .083, 0],
            [.14, .243, .104, .032],
            [.021, .007, .116, .047],
            [0, 0, .012, .026]
        ]

    def get_connection_probability(self, source, target):
        assert source in self._layers
        assert target in self._layers

        source_index = self._layers.index(source)
        target_index = self._layers.index(target)

        return self.probabilities[source_index][target_index]


def check_all_kernels():
    """
    Prints kernel width estimates for all feedforward cortico-cortical connections.
    This takes something like 20 minutes to run.
    """
    data = Data()
    cortical_areas = [area for area in data.get_areas() if not area == 'LGNd']
    for target_area in cortical_areas:
        for source_area in cortical_areas:
            if data.get_hierarchical_level(source_area) < data.get_hierarchical_level(target_area):
                for target_layer in data.get_layers():
                    for source_layer in data.get_layers():
                        print('{}{}-{}{} kernel {} micrometers'.format(
                            source_area,
                            source_layer,
                            target_area,
                            target_layer,
                            data.get_kernel_width(source_area, source_layer, target_area, target_layer)
                        ))


if __name__ == '__main__':
    data = Data()
    # print((data.get_num_neurons('VISp', '4')/37)**.5)
    print(data.get_hit_rate_width('4', '4'))
    print(data.get_hit_rate_width('4', '2/3'))
    print(data.get_hit_rate_width('2/3', '5'))


    # for area in ['VISrl', 'VISli', 'VISpor']:
    #     print(area)
    #     print(data.get_num_neurons(area, '2/3'))
    #     print(data.get_num_neurons(area, '4'))
    #     print(data.get_num_neurons(area, '5'))

    # check_all_kernels()
    # data = Data()
    # print(data.get_kernel_width('VISp', '2/3', 'VISrl', '4'))
