import copy
import json
import math

import declarations
import random
import numpy
import matplotlib
import csv
from mpl_toolkits import mplot3d

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def perceptron_run(points: [], n: float, cot: int, dim: int, perceptron_type: str, b: float):
    i = 0
    w = []
    for j in range(dim + 1):
        w.append(0.0)
    w = numpy.random.uniform(-1, 1, dim + 1)
    print(w)
    error_min = math.inf
    perceptron: declarations.Perceptron = declarations.Perceptron(n, w)
    w_min = []
    while error_min > 0 and i < cot:
        m = random.randint(0, len(points) - 1)
        h = calculate_excitement(points[m], perceptron.w)
        if perceptron_type == 'step':
            o = numpy.sign(h)
            if o == 0:
                o = 1
            delta_w = calculate_delta_w_step(points[m], perceptron.n, o)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_step(points, new_w)
        elif perceptron_type == 'linear':
            o = h
            delta_w = calculate_delta_w_step(points[m], perceptron.n, o)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_linear(points, new_w)
        elif perceptron_type == 'non-linear-tan':
            o = math.tanh(b * h)
            delta_w = calculate_delta_w_non_linear_tan(points[m], perceptron.n, o, b)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_non_linear(points, new_w)
        elif perceptron_type == 'non-linear-logistic':
            o = 1 / (1 + math.e ** (-2 * b * h))
            delta_w = calculate_delta_w_non_linear_logistic(points[m], perceptron.n, o, b)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_non_linear(points, new_w)
        else:
            o = numpy.sign(h)
            if o == 0:
                o = 1
            delta_w = calculate_delta_w_step(points[m], perceptron.n, o)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_step(points, new_w)

        perceptron.w = new_w
        if error <= error_min:
            error_min = error
            w_min = new_w
        i += 1
    perceptron.w = w_min
    print(f"i: {i}")
    print(f"error: {error_min}")
    return perceptron


def multi_layer_perceptron_run(points: [], n: float, cot: int, dim: int, perceptron_type: str, b: float,
                               layers_count: int, nodes_count: int):
    i = 0
    error_min = math.inf
    layers = []
    # 1 inicializar $
    for i in range(layers_count):
        nodes = []
        for j in range(nodes_count):
            if i == 0:
                w = numpy.random.uniform(-1, 1, dim)
                nodes.append(declarations.Node(n, w))
            else:
                w = numpy.random.uniform(-1, 1, nodes_count)
                nodes.append(declarations.Node(n, w))
        layers.append(declarations.Layer(nodes))

    w = numpy.random.uniform(-1, 1, nodes_count)
    nodes = [declarations.Node(n, w)]
    layers.append(declarations.Layer(nodes))
    multilayer_perceptron = declarations.MultilayerPerceptron(n, layers)

    while error_min > 0 and i < cot:
        # 2 tomar ej al azar del conjunto de entrenamiento y aplciar para al capa 0
        m = random.randint(0, len(points) - 1)
        point = points[m]
        # h = calculate_excitement(points[m], perceptron.w)
        # 3 propagar la entrada hasta la capa de salida
        layers = multilayer_perceptron.layers
        for i in range(len(layers)):
            layer = layers[i]
            array = []
            for node in layer.nodes:
                if i == 0:
                    h = calculate_excitement(point, node.w) # TODO: fixme :)
                else:
                    h = calculate_excitement(layers[i - 1].point, node.w)
                array.append(h)
            layer.point.e = array
        # 4 calcular error para la capa de salida
        layer = layers[-1]
        for node in layer.nodes:
            h = layer.point.e[0]
            error = (point.expected_value - calculate_o(h)) * calculate_o_derivative(h)
            node.error = error

        # 5 retropropagar n entre 2 y N
        for i in reversed(range(len(layers) - 1)):
            layer = layers[i]
            for j in range(len(layer.nodes)):
                node = layer.nodes[j]
                h = layer.point.e[j]
                
                error = (point.expected_value - calculate_o(h)) * calculate_o_derivative(h)
                node.error = error

        # 6 actualizar los $ de las conexiones
        # 7 calcular el error
        i += 1


def calculate_o(h: float):
    return h


def calculate_o_derivative(h: float):
    return 1


def calculate_excitement(point: declarations.Point, w: []):
    excitement = 0
    for i in range(len(w)):
        excitement += (point.e[i] * w[i])

    return excitement


def calculate_delta_w_step(point: declarations.Point, n: float, o: float):
    delta_w = []
    wi = n * (point.expected_value - o)
    for ei in point.e:
        delta_w.append(wi * ei)

    return delta_w


def calculate_delta_w_linear(points: [], n: float, dim: int, w: []):
    delta_w = []
    for j in range(dim + 1):
        delta_w.append(0.0)

    for point in points:
        h = calculate_excitement(point, w)
        wi = n * (point.expected_value - h)
        for i in range(len(point.e)):
            delta_w[i] += wi * point.e[i]

    return delta_w


def calculate_delta_w_non_linear_tan(point: declarations.Point, n: float, o: float, b: float):
    # delta_w = []
    # for j in range(dim + 1):
    #    delta_w.append(0.0)
    #
    # for point in points:
    #    h = calculate_excitement(point, w)
    #    o = math.tanh(b * h)
    #    wi = n * (point.expected_value - o) * b * (1 - o)
    #    for i in range(len(point.e)):
    #        delta_w[i] += wi * point.e[i]
    #
    # return delta_w
    delta_w = []
    wi = n * (point.normalized_expected_value - o) * b * (1 - o)
    for ei in point.e:
        delta_w.append(wi * ei)

    return delta_w


def calculate_delta_w_non_linear_logistic(point: declarations.Point, n: float, o: float, b: float):
    # delta_w = []
    # for j in range(dim + 1):
    #    delta_w.append(0.0)
    #
    # for point in points:
    #    h = calculate_excitement(point, w)
    #    o = 1 / (1 + math.e ** (-2*b*h))
    #    wi = n * (point.expected_value - o) * 2 * b * o * (1 - o)
    #    for i in range(len(point.e)):
    #        delta_w[i] += wi * point.e[i]
    #
    # return delta_w
    delta_w = []
    wi = n * (point.normalized_expected_value - o) * 2 * b * o * (1 - o)
    for ei in point.e:
        delta_w.append(wi * ei)

    return delta_w


def calculate_w(delta_w: [], w: []):
    return numpy.sum([w, delta_w], axis=0)


def calculate_error_step(points: [], w: []):
    error = 0
    for point in points:
        h = calculate_excitement(point, w)
        o = numpy.sign(h)
        if o == 0:
            o = 1.0
        error += abs(point.expected_value - o)

    return error


def calculate_error_linear(points: [], w: []):
    error = 0
    for point in points:
        h = calculate_excitement(point, w)
        # print(f"h: {h}, point: {point}")
        error += (point.expected_value - h) ** 2

    return error * 1 / 2


def calculate_error_non_linear(points: [], w: []):
    error = 0
    for point in points:
        h = calculate_excitement(point, w)
        # print(f"h: {h}, point: {point}")
        error += math.pow(point.normalized_expected_value - h, 2)

    return error * 1 / 2


def run(cot=1000, n=0.1, x=None, y=None, b=1, normalization='scale', perceptron_type='linear'):
    config_file = open("config.json")
    config_data = json.load(config_file)
    if not config_data['config_by_code']:
        cot = config_data["cot"]
        n = config_data["n"]
        x = config_data["x"]
        y = config_data["y"]
        b = config_data["b"]
        normalization = config_data['normalization_type']
        perceptron_type = config_data["perceptron_type"]

    point_array = []
    training_set = []
    evaluation_set = []
    dim = 0
    if perceptron_type == 'step':
        for i in range(len(x)):
            arr = [1]
            for p in x[i]:
                arr.append(p)
            point = declarations.Point(arr, y[i])
            point_array.append(point)
            dim = len(x[0])
        training_set = point_array
        evaluation_set = point_array
    elif perceptron_type == 'linear' or perceptron_type == 'non-linear-tan' or perceptron_type == 'non-linear-logistic':
        file = open('TP2-ej2-conjunto.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)
        dim = len(header) - 1
        min_expected_value = math.inf
        max_expected_value = 0
        sum_values = 0
        # TODO check
        for row in csvreader:
            arr = [1]
            expected_value = 0
            for i in range(len(row)):
                if i == len(row) - 1:
                    expected_value = float(row[i])
                    sum_values += expected_value
                    if expected_value < min_expected_value:
                        min_expected_value = expected_value
                    if expected_value > max_expected_value:
                        max_expected_value = expected_value
                else:
                    arr.append(float(row[i]))
            point = declarations.Point(arr, expected_value)
            point_array.append(point)
        r = max_expected_value - min_expected_value
        mean = sum_values / len(point_array)
        sum_dev_squared = 0
        for point in point_array:
            sum_dev_squared += (point.expected_value - mean) ** 2
        standard_deviation = math.sqrt(sum_dev_squared / len(point_array))
        for i in range(len(point_array)):
            if normalization == 'scale':
                if perceptron_type == 'non-linear-tan':
                    point_array[i].normalized_expected_value = 2 * (
                            point_array[i].expected_value - min_expected_value) / r - 1
                if perceptron_type == 'non-linear-logistic':
                    point_array[i].normalized_expected_value = (point_array[i].expected_value - min_expected_value) / r
            if normalization == 'z-score':
                point_array[i].normalized_expected_value = (point_array[i].expected_value - mean) / standard_deviation
            if i < len(point_array) - 1:
                training_set.append(point_array[i])
            else:
                evaluation_set.append(point_array[i])

    perception = perceptron_run(training_set, n, cot, dim, perceptron_type, b)

    print(perception)
    if perceptron_type == 'step':
        error = calculate_error_step(evaluation_set, perception.w)
    if perceptron_type == 'linear':
        error = calculate_error_linear(evaluation_set, perception.w)
    if perceptron_type == 'non-linear-tan' or perceptron_type == 'non-linear-logistic':
        error = calculate_error_non_linear(evaluation_set, perception.w)
    print(f"evaluation error: {error}")

    if perceptron_type == 'step':
        print(f"y = {perception.w[1] / (-1 * perception.w[2])} * x + {perception.w[0] / (-1 * perception.w[2])}")

        x = numpy.linspace(-5, 5, 100)
        div = (-1 * perception.w[2])
        y = perception.w[1] / div * x + perception.w[0] / div
        plt.plot(x, y, '-r')

        for point in point_array:
            color = 'red'
            if point.expected_value == -1:
                color = 'black'
            plt.scatter(point.e[1], point.e[2], color=color)
        plt.show()

    return perception, error


if __name__ == "__main__":
    run()
