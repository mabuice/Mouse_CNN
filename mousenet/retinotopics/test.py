import pickle
import shapely

def test_load_results():
	with open("retinomap.pkl", "rb") as file:
		results = pickle.load(file)

	assert len(results) == 7 #for num areas
	for area in results:
		assert type(area) == tuple #((area, (optimized_polygon, max_iou)), normalized_polygon)
		assert type(area[1]) == shapely.geometry.polygon.Polygon #normalized_polygon


def test_normalized_polygon_coordinates():
	 with open("retinomap.pkl", "rb") as file:
		results = pickle.load(file)

	for area in results:
		normalized_polygon = area[1]
		x, y = normalized_polygon.exterior.coords.xy
		x, y = list(x), list(y)
		assert all([i <= 64 and i>= 0 for i in x])
		assert all([i <= 64 and i>= 0 for i in y])
