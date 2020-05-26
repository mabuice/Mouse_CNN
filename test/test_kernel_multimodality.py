import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mouse_cnn.voxel import VoxelModel, is_multimodal_or_eccentric, fit_image, get_gaussian_fit
from mouse_cnn.voxel import get_fraction_peak_at_center_of_mass, get_multimodal_depth_fraction
from mouse_cnn.flatmap import FlatMap


def save_example_kernels():
    np.random.seed(0)

    source_names = [
        'VISp2/3', 'VISp2/3', 'VISp2/3', 'VISp2/3', 'VISp2/3',
        'VISp5', 'VISp5', 'VISp5', 'VISp5', 'VISp5',
        'VISl2/3', 'VISrl2/3', 'VISli2/3', 'VISpl2/3', 'VISal2/3',
        'VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5'
    ]
    target_names = [
        'VISl4', 'VISrl4', 'VISli4', 'VISpl4', 'VISal4',
        'VISl4', 'VISrl4', 'VISli4', 'VISpl4', 'VISal4',
        'VISpor4', 'VISpor4', 'VISpor4', 'VISpor4', 'VISpor4',
        'VISpor4', 'VISpor4', 'VISpor4', 'VISpor4', 'VISpor4'
    ]

    voxel_model = VoxelModel()
    flatmap = FlatMap.get_instance()

    weights = []
    positions_2d = []
    target_voxel_indices = []
    for source_name, target_name in zip(source_names, target_names):
        print('processing {}->{}'.format(source_name, target_name))

        p = voxel_model.get_positions(source_name)
        positions_2d.append([flatmap.get_position_2d(position) for position in p])

        w = voxel_model.get_weights(source_name, target_name)
        ind = np.random.choice(len(w), 5, replace=False).astype(int)
        print(ind)
        target_voxel_indices.append(ind)
        weights.append([w[i] for i in ind])

    with open('example_kernels.pkl', 'wb') as file:
        data = {
            'source_names': source_names,
            'target_names': target_names,
            'target_voxel_indices': target_voxel_indices,
            'weights': weights,
            'positions_2d': positions_2d
        }
        pickle.dump(data, file)


def load_example_kernels():
    with open('example_kernels.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def multimodal_labels():
    """
    :return: Manually-defined labels for whether each of the example kernels is multimodal.
    """

    # I went through these twice. There were two disagreements, which I rechecked. These
    # are marked 'AMBIGUOUS' below, because they could reasonably be categorized either way.
    # They are labelled True to be conservative. Clear but weak True are marked
    # 'borderline'.

    return [
        [False, True, False, False, False], #0
        [False, True, False, False, False],
        [True, False, False, False, False], # first wraps around
        [True, False, False, False, False], # slight on third
        [False, False, False, True, False],
        [True, True, False, True, False], #5 2nd borderline
        [True, True, False, False, True], #2nd weak, 5th weak
        [False, False, False, True, False],
        [True, True, False, False, False],
        [False, True, False, True, True], # 5th borderline
        [False, False, True, False, False], #10 3rd is AMBIGUOUS (wraps around edge)
        [True, False, False, False, False],
        [True, False, False, False, False], # first is AMBIGUOUS (slight depression in middle)
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False], #15
        [False, False, False, True, False], # 4th borderline
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]
    ]


if __name__ == '__main__':
    save_example_kernels()
    data = load_example_kernels()

    gf = []
    mdf = []
    fpc = []
    lab = []
    images = []
    for weights, positions_2d, labels in zip(data['weights'], data['positions_2d'], multimodal_labels()):
        print(labels)
        for w, l in zip(weights, labels):
            image = fit_image(w, positions_2d)

            gf.append(get_gaussian_fit(image))

            images.append(image)
            mdf.append(get_multimodal_depth_fraction(image))
            fpc.append(get_fraction_peak_at_center_of_mass(image))
            lab.append(l)

    for i in range(len(lab)):
        if lab[i] and fpc[i] > .6:
            plt.imshow(images[i])
            plt.title('arguably multimodal')
            plt.show()

    multimodal = lab
    unimodal = [not l for l in lab]
    mdf = np.array(mdf)
    fpc = np.array(fpc)
    gf = 10000*np.array(gf)

    plt.scatter(mdf[unimodal], fpc[unimodal], c='g')
    plt.scatter(mdf[multimodal], fpc[multimodal], c='r')
    plt.xlabel('multimodal depth fraction')
    plt.ylabel('fraction peak at centroid')
    plt.legend(('unimodal', 'multimodal'))
    plt.show()

