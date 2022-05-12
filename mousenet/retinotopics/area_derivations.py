import pandas as pd
import numpy as np
from shapely.geometry import box, Polygon, LinearRing
import numpy as np
from scipy.spatial import ConvexHull
from shapely.affinity import scale
import itertools

import matplotlib.pyplot as plt
import geopandas as gpd


def iou(A, B):
    """
    Ratio of the intersection of two shapely polygons with their union
    
    :param A: A shapely polygon
    :param B: A shapely polygon

    Returns
    -------
        float: ratio
    """
    if type(A) != Polygon:
        A = Polygon(A)
    if type(B) != Polygon:
        B = Polygon(B)

    return (
        A.intersection(B).area /
        A.union(B).area
    )

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates

    Returns
        4x2 array: the bounding rectangle
    -------

    """
    if type(points) == Polygon:
        points = np.array(points.exterior.xy).T

    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def get_best_fit_rect(area, step=5):
    area_df = pd.read_csv(f"digitized_data/{area}.csv", header=None)
    area_polygon = Polygon(area_df.values)

    x1 = np.arange(area_df[0].min(), area_df[0].max(), step)
    x2 = np.arange(area_df[0].max(), area_df[0].min(), -1*step)
    y1 = np.arange(area_df[1].min(), area_df[1].max(), step)
    y2 = np.arange(area_df[1].max(), area_df[1].min(), -1*step)

    possible_rectangles = list(itertools.product(x1, x2, y1, y2))

    best_rect = possible_rectangles[0]
    best_iou = 0
    for r in possible_rectangles:
        candidate = Polygon(np.array([
            [r[0], r[2]], [r[0], r[3]], 
            [r[1], r[3]], [r[1], r[2]]])
        )

        current_iou = iou(candidate, area_polygon)
        if current_iou > best_iou:
            best_iou = current_iou
            best_rect = candidate

    return best_rect, best_iou