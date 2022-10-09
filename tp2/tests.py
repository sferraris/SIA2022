import numpy
from numpy import split


def tests():
    inner_layers = 2
    nodes_count = 2
    dim = 2
    output_nodes = 1


    weights = {}
    for j in range(inner_layers):
        weights[j] = []
        for i in range(nodes_count):
            weights[j].append(numpy.random.uniform(-1, 1, size=(dim + 1)))

    weights[inner_layers] = []
    for i in range(output_nodes):
        weights[inner_layers].append(numpy.random.uniform(-1, 1, size=(dim + 1)))

    point_dictionary = {}
    points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    k = 3

    point_length = int(len(points) / k)
    indexes = list(range(len(points)))
    numpy.random.shuffle(indexes)

    idx_counter = 0
    for idx in range(k):
        point_dictionary[idx] = []
        for k_idx in range(point_length):
            point_dictionary[idx].append(points[indexes[idx_counter]])
            idx_counter += 1


    arr1 = []
    arr2 = [1, 2, 3]
    arr1 += arr2
    arr1 += points
    print(arr1)

    print(point_dictionary)

if __name__ == "__main__":
    tests()
