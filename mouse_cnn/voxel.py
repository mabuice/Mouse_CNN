import numpy as np
import pickle
from mcmodels.core import VoxelModelCache
from mouse_cnn.flatmap import FlatMap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy.ndimage.filters import gaussian_filter
import matplotlib.path as path
from sklearn.kernel_ridge import KernelRidge
from skimage.morphology import watershed, h_minima, h_maxima
from scipy.optimize import curve_fit

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
    # note: gamma scaling should be local to each target
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

        c = 0
        for target_voxel in range(len(weights)):
            # if c == 3:
            #     with open('foo.pkl', 'wb') as f:
            #         pickle.dump((weights[target_voxel], positions_2d), f)
            #     # assert False
            c += 1

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


def fit_image(weights, positions_2d):
    """
    :param weights: connectivity weights for source voxels
    :param positions_2d: flatmap positions of source voxels
    :return: approximation of connection density on a grid
    """
    positions_2d = np.array(positions_2d)

    range_x = [np.min(positions_2d[:,0]), np.max(positions_2d[:,0])]
    range_y = [np.min(positions_2d[:,1]), np.max(positions_2d[:,1])]

    n_steps = 20
    x = np.linspace(range_x[0], range_x[1], n_steps)
    y = np.linspace(range_y[0], range_y[1], n_steps)

    X, Y = np.meshgrid(x, y)

    regression = KernelRidge(alpha=1, kernel='rbf')
    regression.fit(positions_2d, weights)

    coords = np.zeros((n_steps**2, 2))
    coords[:,0] = X.flatten()
    coords[:,1] = Y.flatten()

    prediction = regression.predict(coords)
    prediction = np.reshape(prediction, (n_steps, n_steps))

    hull = ConvexHull(positions_2d)
    v = np.concatenate((hull.vertices, [hull.vertices[0]]))

    p = path.Path([(positions_2d[i,0], positions_2d[i,1]) for i in v])
    inside = p.contains_points(coords)
    outside = [not x for x in inside]

    lowest = np.min(prediction)
    highest = np.max(prediction)

    prediction = np.reshape(prediction, n_steps**2)
    prediction[outside] = lowest
    prediction[prediction < lowest + 0.2*(highest-lowest)] = lowest
    prediction = np.reshape(prediction, (n_steps, n_steps))

    prediction = gaussian_filter(prediction, 1, mode='nearest')
    prediction = prediction - np.min(prediction)

    return prediction


def get_multimodal_depth_fraction(image):
    lowest = np.min(image)
    highest = np.max(image)

    step = .02
    for f in np.arange(0, 1+step, step):
        maxima = h_maxima(image, f * (highest - lowest))
        s = np.sum(maxima)

        # I don't know why the second condition is needed below, but sometimes with a unimodal
        # image, s is a matrix of all ones
        if s == 1 or s == maxima.size:
            return f

    return 1


def get_fraction_peak_at_centroid(image):
    image = image - np.min(image)
    cx0, cx1 = get_centroid(image)
    value_at_centroid = image[int(round(cx0)), int(round(cx1))]
    return value_at_centroid / np.max(image)


def get_centroid(image):
    X1, X0 = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    total = np.sum(image)
    cx0 = np.sum(X0 * image) / total
    cx1 = np.sum(X1 * image) / total
    return cx0, cx1


def get_gaussian_fit(image):
    X1, X0 = np.meshgrid(range(image.shape[0]), range(image.shape[1]))
    X = np.concatenate((X0.flatten()[:,None], X1.flatten()[:,None]), axis=1)

    def gaussian(peak, mean, covariance):
        ic = np.linalg.inv(covariance)
        offsets = (X - mean)
        distances = [np.dot(offset, np.dot(ic, offset.T))/2 for offset in offsets]
        return peak * np.exp(-np.array(distances)/2)

    def f(x, peak, m1, m2, cov11, cov12, cov22):
        return gaussian(peak, [m1, m2], [[cov11, cov12], [cov12, cov22]])

    scale = np.max(image)
    rescaled_image = image / scale

    s = image.shape[0]
    cx0, cx1 = get_centroid(image)
    p0 = [np.max(rescaled_image), cx0, cx1, 3, 0, 3]
    lower_bounds = [0, 0, 0, 1, 0, 1]
    upper_bounds = [np.max(rescaled_image), s, s, s/2, s/2, s/2]
    popt, pcov = curve_fit(f, X, rescaled_image.flatten(), bounds=(lower_bounds, upper_bounds), p0=p0)
    popt[0] = popt[0]*scale

    fit = f(None, *popt)

    difference = fit.reshape(image.shape) - image
    rmse = np.mean(difference**2)**.5
    # print('RMSE: {}'.format(rmse))

    # print(p0)
    # print(popt)
    # plt.subplot(1,2,1)
    # plt.imshow(image)
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(fit.reshape(image.shape))
    # plt.colorbar()
    # plt.show()

    return rmse


def is_multimodal(weights, positions_2d):
    """
    :param weights: connectivity weights for source voxels
    :param positions_2d: flatmap positions of source voxels
    :return: True if weights have multiple dense regions, False if single dense region
    """
    image = fit_image(weights, positions_2d)
    return get_fraction_peak_at_centroid(image) < .6


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


def flatmap_weights(positions_2d, weights):
    rel_weights = weights / max(weights)
    for position, rel_weight in zip(positions_2d, rel_weights):
        color = [rel_weight, 0, 1 - rel_weight, .5]
        plt.scatter(position[0], position[1], c=[color])
    plt.xticks([]), plt.yticks([])


if __name__ == '__main__':
    # vm = VoxelModel()
    # weights = vm.get_weights(source_name='VISp2/3', target_name='VISpm4')

    t = Target('VISpm', '4')
    t.set_gamma()
    print('VISp2/3->VISpm4 kernel width estimate: {}'.format(t.get_kernel_width_mm('VISp2/3')))
    print(t)

    # with open('foo.pkl', 'rb') as f:
    #     weights, positions_2d = pickle.load(f)
    #
    # multimodal = is_multimodal(weights, positions_2d)
    # print('multimodal={}'.format(multimodal))

