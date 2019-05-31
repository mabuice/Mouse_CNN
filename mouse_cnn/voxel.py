import numpy as np
import pickle
from mcmodels.core import VoxelModelCache
from mouse_cnn.flatmap import FlatMap


"""
Code for estimating density profiles of inter-area connections from voxel model
of mouse connectome (Knox et al. 2019).
"""


class VoxelModel():
    # we make a shared instance because the model's state doesn't change
    # but it takes several seconds to instantiate, so we only want to do it once
    _instance = None

    def __init__(self):
        cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        self.source_mask = cache.get_source_mask()
        self.source_keys = self.source_mask.get_key(structure_ids=None)

        with open('data_files/voxel-connectivity-weights.pkl', 'rb') as file:
            self.weights = pickle.load(file)
        with open('data_files/voxel-connectivity-nodes.pkl', 'rb') as file:
            self.nodes = pickle.load(file)

        self.structure_tree = cache.get_structure_tree()

    def get_weights(self, source_name='VISp2/3', target_name='VISpm4'):
        pre_id = self.structure_tree.get_id_acronym_map()[source_name]
        post_id = self.structure_tree.get_id_acronym_map()[target_name]

        pre_indices = []
        post_indices = []
        for i in range(len(self.source_keys)):
            if self.structure_tree.structure_descends_from(self.source_keys[i], pre_id):
                pre_indices.append(i)
            if self.structure_tree.structure_descends_from(self.source_keys[i], post_id):
                post_indices.append(i)

        weights_by_target_voxel = []
        for pi in post_indices:
            w = np.dot(self.weights[pre_indices,:], self.nodes[:,pi])
            weights_by_target_voxel.append(w)
        return weights_by_target_voxel

    def get_positions(self, source_name='VISp2/3'):
        pre_id = self.structure_tree.get_id_acronym_map()[source_name]
        mask_indices = np.array(self.source_mask.mask.nonzero())

        pre_positions = []
        for i in range(len(self.source_keys)):
            if self.structure_tree.structure_descends_from(self.source_keys[i], pre_id):
                pre_positions.append(mask_indices[:, i])

        return pre_positions


    @staticmethod
    def get_instance():
        """
        :return: Shared instance of VoxelModel
        """
        if VoxelModel._instance is None:
            VoxelModel._instance = VoxelModel()
        return VoxelModel._instance


areas = ['VISp', 'VISpm']
layers = ['2/3', '4', '5']


class Target():
    def __init__(self, area, layer, external_in_degree=1000):
        """
        :param area: name of area
        :param layer: name of layer
        :param external_in_degree: Total neurons providing feedforward input to average
            neuron from other cortical areas.
        """
        self.target_area = area
        self.target_name = area + layer
        self.e = external_in_degree

        self.voxel_model = VoxelModel.get_instance()

        self.gamma = None # scale factor for total inbound voxel weight -> extrinsic in-degree

        self.source_names = None # list of possible extrinsic source area / layers
        self.mean_totals = None # mean of total inbound weight across target voxels for each source

    def _set_external_sources(self):
        """
        :return: Names of sources (area, layer) that may project to this target,
            excluding other layers in the same area
        """
        self.source_names = []
        for area in areas:
            if not area == self.target_area:
                for layer in layers:
                    self.source_names.append(area + layer)

    def _set_mean_total_weights(self):
        if self.source_names is None:
            self._set_external_sources()

        self.mean_totals = []
        for source_name in self.source_names:
            self.mean_totals.append(self._find_mean_total_weight(source_name))

    def _find_mean_total_weight(self, source_name):
        """
        :param source_name: source area/layer name, e.g. VISp2/3
        :return: mean of the total voxel-model weight inbound to each target voxel,
            from the source
        """
        weights = self.voxel_model.get_weights(source_name, self.target_name)
        totals = [np.sum(w) for w in weights]
        return np.mean(totals)

    def set_gamma(self):
        if self.mean_totals is None:
            self._set_mean_total_weights()

        print(self.mean_totals)
        self.gamma = self.e / np.sum(self.mean_totals)

    def get_n_external_inputs_for_source(self, source_name):
        if self.gamma is None:
            self.set_gamma()

        assert source_name in self.source_names

        index = self.source_names.index(source_name)
        return self.mean_totals[index] * self.gamma

    def get_peak_connection_probability(self, source_name, cortical_magnification, beta, c_out):
        """
        TODO: maybe this method belongs somewhere else as it's network-centric.

        :param source_name: source area/layer name
        :param cortical_magnification: mm cortex per degree visual angle
        :param beta: degrees visual angle per pixel of source feature map
        :param c_out: number of feature maps in source
        :return: hit rate at centre of kernel
        """
        e_ij = self.get_n_external_inputs_for_source(source_name)
        d_w = self.get_kernel_width_pixels(source_name, cortical_magnification, beta)
        d_p = e_ij / (c_out * 2*np.pi * d_w**2)
        return d_p

    def get_kernel_width_pixels(self, source_name, cortical_magnification, beta):
        """
        TODO: maybe this method belongs somewhere else as it's network-centric.

        :param source_name: source area/layer name
        :param cortical_magnification: mm cortex per degree visual angle
        :param beta: degrees visual angle per pixel of source feature map
        :return: width (sigma) of Gaussian kernel approximation in feature-map pixels
        """
        return self.get_kernel_width_degrees(source_name, cortical_magnification) / beta

    def get_kernel_width_degrees(self, source, cortical_magnification):
        """
        :param source: source area/layer name
        :param cortical_magnification: mm cortex per degree visual angle
        :return: width (sigma) of Gaussian kernel approximation in degrees visual angle
        """
        return self.get_kernel_width_mm(source) / cortical_magnification

    def get_kernel_width_mm(self, source_name):
        """
        :param source_name: source area/layer name
        :return: sigma of Gaussian approximation of mean input kernel
        """
        sigmas = []

        weights = self.voxel_model.get_weights(source_name, self.target_name) # target voxel by source voxel
        positions = self.voxel_model.get_positions(source_name) # source voxel by 3

        flatmap = FlatMap.get_instance()
        positions_2d = [flatmap.get_position_2d(position) for position in positions] # source voxel by 2

        for target_voxel in range(len(weights)):
            if not is_multimodal(weights[target_voxel], positions_2d):
                sigmas.append(find_radius(weights[target_voxel], positions_2d))

        return np.mean(sigmas)

    def __str__(self):
        result = '{} gamma={}'.format(self.target_name, self.gamma)
        if self.source_names:
            for source, mean_total in zip(self.source_names, self.mean_totals):
                result += '\n{} mean-total weight: {:.3}  external inputs: {:.4}'.format(
                    source, mean_total, self.get_n_external_inputs_for_source(source))
        return result


def is_multimodal(weights, positions_2d):
    """
    :param weights: connectivity weights for source voxels
    :param positions_2d: flatmap positions of source voxels
    :return: True if weights have multiple dense regions, False if single dense region
    """
    #TODO: implement - use watershed algorithm
    return False

def find_radius(weights, positions_2d):
    #TODO: deconvolve from model blur and flatmap blur
    positions_2d = np.array(positions_2d)
    total = sum(weights)
    centroid_x = np.sum(weights * positions_2d[:,0]) / total
    centroid_y = np.sum(weights * positions_2d[:,1]) / total
    offset_x = positions_2d[:,0] - centroid_x
    offset_y = positions_2d[:,1] - centroid_y
    square_distance = offset_x**2 + offset_y**2
    return (np.sum(weights * square_distance) / total)**.5


if __name__ == '__main__':
    # vm = VoxelModel()
    # weights = vm.get_weights(source_name='VISp2/3', target_name='VISpm4')

    t = Target('VISpm', '4')
    t.set_gamma()
    print('VISp2/3->VISpm4 kernel width estimate: {}'.format(t.get_kernel_width_mm('VISp2/3')))
    print(t)


