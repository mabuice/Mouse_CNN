import numpy as np
import pickle
from mcmodels.core import VoxelModelCache
from mouse_cnn.flatmap import FlatMap
from mouse_cnn.data import Data
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage.filters import gaussian_filter
import matplotlib.path as path
from sklearn.kernel_ridge import KernelRidge
from skimage.morphology import h_maxima
from scipy.optimize import curve_fit

"""
Code for estimating density profiles of inter-area connections from voxel model
of mouse connectome (Knox et al. 2019).
"""


class VoxelModel():
    # we make a shared instance because the model's state doesn't change
    # but it takes several seconds to instantiate, so we only want to do it once
    _instance = None

    def __init__(self, data_folder='data_files/'):
        cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
        self.source_mask = cache.get_source_mask()
        self.source_keys = self.source_mask.get_key(structure_ids=None)

        # new version (runs slow)
        # self.weights = cache.get_weights()
        # self.nodes = cache.get_nodes()

        # old version (runs fast)
        with open(data_folder + '/voxel-connectivity-weights.pkl', 'rb') as file:
            self.weights = pickle.load(file)
        with open(data_folder + '/voxel-connectivity-nodes.pkl', 'rb') as file:
            self.nodes = pickle.load(file)

        self.structure_tree = cache.get_structure_tree()

    def get_weights(self, source_name, target_name):
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

    def get_positions(self, source_name):
        pre_id = self.structure_tree.get_id_acronym_map()[source_name]
        mask_indices = np.array(self.source_mask.mask.nonzero())

        pre_positions = []
        for i in range(len(self.source_keys)):
            if self.structure_tree.structure_descends_from(self.source_keys[i], pre_id):
                pre_positions.append(mask_indices[:, i])

        return pre_positions

    @staticmethod
    def get_instance(data_folder='data_files/'):
        """
        :return: Shared instance of VoxelModel
        """
        if VoxelModel._instance is None:
            VoxelModel._instance = VoxelModel(data_folder=data_folder)
        return VoxelModel._instance


class Target():
    """
    A model of the incoming inter-area connections to a target area / layer, including
    gaussian kernel widths and peak hit rates for each inbound connection. This model
    does not deal with inter-laminar connections within an area, which are based on
    different data sources (not the voxel model).

    An important property of the model is the variable "gamma", which maps voxel-model
    connection weight, w, to in-degree, d. Specifically, d = gamma w. Gamma is
    taken to be a property of a target area/layer that may be different for different
    targets. Allowing gamma to vary between targets simplifies its estimate.
    A more constrained estimate could be obtained by assuming it is shared across all
    targets. On the other hand, the weights measure axon density, which may not have
    exactly the same relationship with numbers of connections for all targets. Here
    we assume only that it is constant for all sources within a target. This may not
    be true either, but it allows us to estimate numbers of connections from voxel weights.
    """

    def __init__(self, area, layer, external_in_degree, data_folder='data_files/'):
        """
        :param area: name of area
        :param layer: name of layer
        :param external_in_degree: Total neurons providing feedforward input to average
            neuron, from other cortical areas.
        """
        self.data_folder=data_folder
        self.target_area = area
        self.target_name = area + layer
        self.e = external_in_degree

        self.voxel_model = VoxelModel.get_instance(data_folder=data_folder)
        self.num_voxels = len(self.voxel_model.get_positions(self.target_name))

        self.gamma = None # scale factor for total inbound voxel weight -> extrinsic in-degree

        self.source_names = None # list of possible extrinsic source area / layers
        self.mean_totals = None # mean of total inbound weight across *target* voxels for each source

    def _set_external_sources(self):
        """
        :return: Names of sources (area, layer) that may project to this target,
            including only lower areas in the visual hierarchy
        """
        self.source_names = []
        data = Data(data_folder=self.data_folder)
        for area in data.get_areas():
            if data.get_hierarchical_level(area) < data.get_hierarchical_level(self.target_area):
                if 'LGN' not in area: #TODO: handle LGN->VISp as special case
                    for layer in data.get_layers():
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

        self.gamma = self.e / np.sum(self.mean_totals)

    def get_n_external_inputs_for_source(self, source_name):
        if self.gamma is None:
            self.set_gamma()

        assert source_name in self.source_names

        index = self.source_names.index(source_name)
        return self.mean_totals[index] * self.gamma

    # def get_kernel_width_degrees(self, source, cortical_magnification):
    #     """
    #     :param source: source area/layer name
    #     :param cortical_magnification: mm cortex per degree visual angle
    #     :return: width (sigma) of Gaussian kernel approximation in degrees visual angle
    #     """
    #     return self.get_kernel_width_mm(source) / cortical_magnification

    def get_kernel_width_mm(self, source_name, plot=False):
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
            source = Source(weights[target_voxel], positions_2d)

            if not is_multimodal_or_eccentric(weights[target_voxel], positions_2d):
                sigmas.append(find_radius(weights[target_voxel], positions_2d))
                if plot:
                    flatmap_weights(positions_2d, weights[target_voxel])
                    plt.title('sigma: {} peak to border: {}'.format(sigmas[-1], source.peak_border_distance))
                    plt.show()

        # plt.hist(sigmas, 20)
        # plt.xlabel('sigma'), plt.ylabel('count')
        # plt.show()

        return np.mean(sigmas)

    def flatmap_full_source_layer(self, layer, target_voxel):
        visual_areas = ['VISp', 'VISl', 'VISrl', 'VISli', 'VISpl', 'VISal', 'VISpor']
        visual_areas.extend(['VISpm', 'VISa', 'VISam'])

        flatmap = FlatMap.get_instance()

        all_weights = []
        all_positions = []
        source_areas = []
        max_weight = 0
        for area in visual_areas:
            source_name = area + layer
            weights = self.voxel_model.get_weights(source_name, self.target_name) # target voxel by source voxel
            positions = self.voxel_model.get_positions(source_name) # source voxel by 3
            positions_2d = [flatmap.get_position_2d(position) for position in positions] # source voxel by 2
            positions_2d = np.array(positions_2d)

            m = np.max(np.array(weights))
            if m > max_weight:
                max_weight = m

            all_weights.append(weights)
            all_positions.append(positions_2d)
            source_areas.append(area)

        for weights, positions_2d, source_area in zip(all_weights, all_positions, source_areas):
            flatmap_weights(positions_2d, weights[target_voxel])
            hull = ConvexHull(positions_2d)
            v = np.concatenate((hull.vertices, [hull.vertices[0]]))
            path = np.array([(positions_2d[i, 0], positions_2d[i, 1]) for i in v])

            print('{} {}'.format(source_area, self.target_name))
            if self.target_area == source_area:
                plt.plot(path[:, 0], path[:, 1], 'r')
                target_position = self.voxel_model.get_positions(self.target_name)[target_voxel]
                target_position_2d = flatmap.get_position_2d(target_position)
                plt.plot(target_position_2d[0], target_position_2d[1], 'rx', markersize=18)
            else:
                plt.plot(path[:, 0], path[:, 1], 'k')

        plt.savefig('L{}-to-{}-voxel{}.png'.format(
            layer.replace('/', ''),
            self.target_name.replace('/', ''),
            target_voxel))
        plt.show()

    def __str__(self):
        result = '{} gamma={}'.format(self.target_name, self.gamma)
        if self.source_names:
            for source, mean_total in zip(self.source_names, self.mean_totals):
                result += '\n{} mean-total weight: {:.3}  external inputs: {:.4}'.format(
                    source, mean_total, self.get_n_external_inputs_for_source(source))
        return result


def get_surface_area_mm2(source_name):
    voxel_model = VoxelModel.get_instance()
    positions = voxel_model.get_positions(source_name)  # source voxel by 3

    flatmap = FlatMap.get_instance()
    positions_2d = [flatmap.get_position_2d(position) for position in positions]  # source voxel by 2
    positions_2d = np.array(positions_2d)

    hull = ConvexHull(positions_2d)
    return hull.volume #hull.area returns the circumference rather than the area


class Source:
    def __init__(self, weights, positions_2d):
        self.weights = weights
        self.positions_2d = np.array(positions_2d)

        self.regression = KernelRidge(alpha=1, kernel='rbf')
        self.regression.fit(positions_2d, weights)

        hull = ConvexHull(positions_2d)
        v = np.concatenate((hull.vertices, [hull.vertices[0]]))
        self.convex_hull = path.Path([(self.positions_2d[i,0], self.positions_2d[i,1]) for i in v])

        self.coords, self.image = self._get_image()

        self.peak = self._find_peak()
        self.peak_border_distance = self._distance_to_border(self.peak)

    def _distance_to_border(self, coords):
        # print('*********')
        # print(self.convex_hull.vertices)
        min_distance = 1e6
        for i in range(len(self.convex_hull.vertices) - 1):
            distance = _distance_to_line_segment(
                coords,
                self.convex_hull.vertices[i],
                self.convex_hull.vertices[i+1])
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def _find_peak(self):
        # rough peak from precomputed image
        max_ind = np.argmax(self.image)
        max_coords = self.coords[max_ind, :]
        # print('max coords: {} val: {}'.format(max_coords, self.image.flatten()[max_ind]))

        # fine peak from local image around rough peak
        range_x = [max_coords[0]-.25, max_coords[0]+.25]
        range_y = [max_coords[1]-.25, max_coords[1]+.25]
        n_steps = 20
        coords = self._get_coords(n_steps, range_x=range_x, range_y=range_y)
        fine_image = self.regression.predict(coords)

        inside = self.convex_hull.contains_points(coords)
        outside = [not x for x in inside]
        fine_image[outside] = 0

        max_ind = np.argmax(fine_image)
        max_coords = coords[max_ind, :]

        # print('max coords: {} val: {}'.format(max_coords, fine_image.flatten()[max_ind]))

        # plt.figure(figsize=(8,3))
        # plt.subplot(1,2,1)
        # plt.imshow(self.image)
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow(np.reshape(fine_image, (n_steps, n_steps)))
        # plt.colorbar()
        # plt.show()

        return max_coords

    def _get_coords(self, n_steps, range_x=None, range_y=None):
        if not range_x:
            range_x = [np.min(self.positions_2d[:, 0]), np.max(self.positions_2d[:, 0])]
        if not range_y:
            range_y = [np.min(self.positions_2d[:, 1]), np.max(self.positions_2d[:, 1])]

        x = np.linspace(range_x[0], range_x[1], n_steps)
        y = np.linspace(range_y[0], range_y[1], n_steps)

        X, Y = np.meshgrid(x, y)

        coords = np.zeros((n_steps**2, 2))
        coords[:,0] = X.flatten()
        coords[:,1] = Y.flatten()
        return coords

    def _get_image(self):
        n_steps = 20
        coords = self._get_coords(n_steps)

        prediction = self.regression.predict(coords)
        prediction = np.reshape(prediction, (n_steps, n_steps))

        inside = self.convex_hull.contains_points(coords)
        outside = [not x for x in inside]

        lowest = np.min(prediction)
        highest = np.max(prediction)

        prediction = np.reshape(prediction, n_steps**2)
        prediction[outside] = lowest
        prediction[prediction < lowest + 0.2*(highest-lowest)] = lowest
        prediction = np.reshape(prediction, (n_steps, n_steps))

        prediction = gaussian_filter(prediction, 1, mode='nearest')
        prediction = prediction - np.min(prediction)

        return coords, prediction


def _distance_to_line_segment(coords, a, b):
    coords, a, b = np.array(coords), np.array(a), np.array(b)
    unit_vector = (b - a) / np.linalg.norm(b - a)
    projection = np.dot(coords - a, unit_vector)

    if projection < 0:
        closest_point = a
    elif projection > 1:
        closest_point = b
    else:
        closest_point = a + projection * unit_vector

    return np.linalg.norm(coords - closest_point)


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


def get_fraction_peak_at_center_of_mass(image):
    image = image - np.min(image)
    cx0, cx1 = get_center_of_mass(image)
    value_at_center_of_mass = image[int(round(cx0)), int(round(cx1))]
    return value_at_center_of_mass / np.max(image)


def get_center_of_mass(image):
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
    cx0, cx1 = get_center_of_mass(image)
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


def is_multimodal_or_eccentric(weights, positions_2d):
    """
    :param weights: connectivity weights for source voxels
    :param positions_2d: flatmap positions of source voxels
    :return: True if weights have multiple dense regions, False if single dense region
    """
    if max(weights) == 0:
        return False
    else:
        image = fit_image(weights, positions_2d)
        return get_fraction_peak_at_center_of_mass(image) < .5


def find_radius(weights, positions_2d):
    #TODO (Stefan): deconvolve from model blur and flatmap blur?
    positions_2d = np.array(positions_2d)
    total = sum(weights)
    if total == 0:
        return 0, 0
    else:
        center_of_mass_x = np.sum(weights * positions_2d[:,0]) / total
        center_of_mass_y = np.sum(weights * positions_2d[:,1]) / total
        offset_x = positions_2d[:,0] - center_of_mass_x
        offset_y = positions_2d[:,1] - center_of_mass_y
        square_distance = offset_x**2 + offset_y**2
        standard_deviation = (np.sum(weights * square_distance) / total)**.5
        # weighted_mean_distance_from_center = np.sum(weights * np.sqrt(square_distance)) / total
        return standard_deviation, total


def flatmap_weights(positions_2d, weights, max_weight=None):
    if max_weight is None:
        max_weight = max(weights)
    rel_weights = weights / max_weight

    for position, rel_weight in zip(positions_2d, rel_weights):
        color = [rel_weight, 0, 1 - rel_weight, .5]
        plt.scatter(position[0], position[1], c=[color])
    # plt.xticks([]), plt.yticks([])


if __name__ == '__main__':
    for area in ['VISl', 'VISrl', 'VISli', 'VISpor']:
        print('\'{}\': {},'.format(area, get_surface_area_mm2(area+'2/3')))

    # vm = VoxelModel()
    # weights = vm.get_weights(source_name='VISp2/3', target_name='VISpm4')

    # t = Target('VISpor', '4', external_in_degree=1000)
    # print('VISl2/3->VISpor4 kernel width estimate: {}'.format(t.get_kernel_width_mm('VISl2/3')))

    # TODO: vast majority peak at border
    # TODO: find multiple peaks in whole visual cortex, keep ones in source area
    # t = Target('VISrl', '4', external_in_degree=1000)
    # print('VISp2/3->VISrl4 kernel width estimate: {}'.format(t.get_kernel_width_mm('VISrl2/3')))

    # t = Target('VISpl', '4', external_in_degree=1000)
    # t.set_gamma()
    # print('VISp2/3->VISpl4 kernel width estimate: {}'.format(t.get_kernel_width_mm('VISp2/3')))
    # print(t)

    # with open('foo.pkl', 'rb') as file:
    #     (rel_weights, positions_2d) = pickle.load(file)
    # print(len(rel_weights))

    # t = Target('VISpl', '4', external_in_degree=1000)
    # print('{} {} voxels'.format(t.target_name, t.num_voxels))
    # t.flatmap_full_source_layer('2/3', 10)

    # source_name = 'VISp2/3'
    # positions = t.voxel_model.get_positions(source_name)  # source voxel by 3
    # weights = t.voxel_model.get_weights(source_name, t.target_name)  # target voxel by source voxel
    #
    # flatmap = FlatMap.get_instance()
    # positions_2d = [flatmap.get_position_2d(position) for position in positions]  # source voxel by 2
    #
    # print(len(weights))
    # print(len(weights[0]))
    # print(len(positions_2d))
    #
    # for target_voxel in range(len(weights)):
    #     flatmap_weights(positions_2d, weights[target_voxel])
    #     plt.show()

    #TODO: is path better than ancestors?
    # path = vm.structure_tree.get_structures_by_id([pre_id])[0]['structure_id_path']
