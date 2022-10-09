import copy
import json
import math

import declarations
import random
import numpy
import matplotlib
import csv
from mpl_toolkits import mplot3d
from declarations import Point

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def perceptron_run(points: [], n: float, cot: int, dim: int, perceptron_type: str, b: float):
    i = 0
    w = []
    for j in range(dim + 1):
        w.append(0.0)
    w = numpy.random.uniform(-1, 1, dim + 1)
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
            error = calculate_error_non_linear(points, new_w, perceptron_type, b)
        elif perceptron_type == 'non-linear-logistic':
            o = 1 / (1 + math.e ** (-2 * b * h))
            delta_w = calculate_delta_w_non_linear_logistic(points[m], perceptron.n, o, b)
            new_w: [] = calculate_w(delta_w, perceptron.w)
            error = calculate_error_non_linear(points, new_w, perceptron_type, b)
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


def multi_layer_perceptron_run_old(points: [], n: float, cot: int, dim: int, b: float,
                                   layers_count: int, nodes_count: int):
    stop_index = 0
    error_min = math.inf
    layers = []
    # 1 inicializar $
    for i in range(layers_count):
        nodes = []
        for j in range(nodes_count):
            if i == 0:
                w = numpy.random.uniform(-1, 1, dim + 1)
                nodes.append(declarations.Node(n, w))
            else:
                w = numpy.random.uniform(-1, 1, nodes_count + 1)
                nodes.append(declarations.Node(n, w))
        layers.append(declarations.Layer(nodes))

    w = numpy.random.uniform(-1, 1, nodes_count)
    nodes = [declarations.Node(n, w)]
    layers.append(declarations.Layer(nodes))
    multilayer_perceptron = declarations.MultilayerPerceptron(n, layers)

    while error_min > 0 and stop_index < cot:
        # 2 tomar ej al azar del conjunto de entrenamiento y aplciar para al capa 0
        m = random.randint(0, len(points) - 1)
        point = points[m]
        # h = calculate_excitement(points[m], perceptron.w)
        # 3 propagar la entrada hasta la capa de salida
        layers = multilayer_perceptron.layers
        for i in range(len(layers)):
            layer = layers[i]
            if i < len(layers):
                array = [1]
            else:
                array = []

            for node in layer.nodes:
                if i == 0:
                    h = calculate_excitement(point, node.w)
                else:
                    h = calculate_excitement(layers[i - 1].point, node.w)
                array.append(h)
                if math.isnan(h):
                    print(f"h is nan -> {array}")
                    exit()
            layer.point.e = array
        # 4 calcular error para la capa de salida
        layer = layers[-1]
        for node in layer.nodes:
            h = layer.point.e[0]
            error = (point.expected_value - calculate_o(h, b)) * calculate_o_derivative(h, b)
            node.error = error

        # 5 retropropagar n entre 2 y N
        for i in reversed(range(len(layers) - 1)):
            layer = layers[i]
            for j in range(len(layer.nodes)):
                node = layer.nodes[j]
                h = layer.point.e[j + 1]
                if i + 1 == len(layers) - 1:
                    top_w = layers[i + 1].nodes[0].w
                    top_e = layers[i + 1].point.e[0]
                else:
                    top_w = layers[i + 1].nodes[j].w
                    top_e = layers[i + 1].point.e[j + 1]

                error_array = calculate_o_derivative(h, b) * numpy.dot(top_w, top_e)

                error = 0
                for e in error_array:
                    error += e

                node.error = error

        # 6 actualizar los $ de las conexiones
        layer = layers[0]
        for j in range(len(layer.nodes)):
            node = layer.nodes[j]
            delta_w = n * numpy.dot(point.e, node.error)

            w_array = []
            for i in range(len(node.w)):
                w_array.append(node.w[i] + delta_w[i])

            node.w = w_array

        for i in range(len(layers) - 2):
            j = i + 1
            node = layer.nodes[j]
            delta_w = n * numpy.dot(layers[i].point.e[j], node.error)
            node.w = numpy.sum([node.w, delta_w], axis=0)

        layer = layers[-1]
        for j in range(len(layer.nodes)):
            node = layer.nodes[j]
            delta_w = n * numpy.dot(layers[-2].point.e[j], node.error)
            node.w = numpy.sum([node.w, delta_w], axis=0)
        # 7 calcular el error
        error = calculate_multi_layer_error(points, multilayer_perceptron, b)
        if error <= error_min:
            error_min = error

        stop_index += 1

    return multilayer_perceptron


def multi_layer_perceptron_run(points: [], n: float, cot: int, dim: int, b: float, inner_layers: int, nodes_count: int,
                               output_nodes: int, momentum: bool, adaptative: bool, adam: bool):
    stop_index = 0
    error_min = math.inf
    alpha = 0.8
    max_error_progress = 6
    alpha_adam = 0.001
    betha1 = 0.9
    betha2 = 0.999
    e = 10e-8
    t = 0
    mt = {}
    weights = init_weights(inner_layers, nodes_count, dim, output_nodes)
    errors = []
    epocas = []
    for i in range(len(weights)):
        mt[i + 1] = []
        for j in range(len(weights[i])):
            mt[i + 1].append([])
            for z in range(len(weights[i][j])):
                mt[i + 1][j].append(0)
    vt = copy.deepcopy(mt)
    mt_corrected = copy.deepcopy(mt)
    vt_corrected = copy.deepcopy(mt)

    min_weights = copy.deepcopy(weights)
    delta_w_dictionary = {}
    error = 0
    error_progress = 0
    last_error = 0
    while error_min > 1E-3 and stop_index < cot:
        indexes = list(range(len(points)))
        numpy.random.shuffle(indexes)
        # random_index = random.randint(0, len(points) - 1)
        epocas.append(stop_index)
        for random_index in indexes:
            t += 1
            point = points[random_index]

            h_dictionary, o_dictionary = p_forward(inner_layers, weights, nodes_count, output_nodes, point, b)

            error_dictionary = p_back(h_dictionary, o_dictionary[inner_layers + 1], inner_layers, weights, nodes_count,
                                      output_nodes, b, point.expected_value)
            if not adam:
                if momentum and stop_index > 0:
                    delta_w_old = copy.deepcopy(delta_w_dictionary)
                    delta_w_dictionary = calculate_delta_w(o_dictionary, error_dictionary, n, inner_layers, nodes_count,
                                                           output_nodes, dim)
                    for i in range(len(delta_w_dictionary)):
                        for j in range(len(delta_w_dictionary[i + 1])):
                            for z in range(len(delta_w_dictionary[i + 1][j])):
                                delta_w_dictionary[i + 1][j][z] += delta_w_old[i + 1][j][z] * alpha
                else:
                    delta_w_dictionary = calculate_delta_w(o_dictionary, error_dictionary, n, inner_layers, nodes_count,
                                                           output_nodes, dim)

                weights = calculate_new_weights(delta_w_dictionary, weights, inner_layers, nodes_count, output_nodes)
            else:
                delta_w_dictionary = calculate_delta_w(o_dictionary, error_dictionary, n, inner_layers, nodes_count,
                                                       output_nodes, dim)
                for i in range(len(delta_w_dictionary)):
                    for j in range(len(delta_w_dictionary[i + 1])):
                        for z in range(len(delta_w_dictionary[i + 1][j])):
                            mt[i + 1][j][z] = betha1 * mt[i + 1][j][z] + (1 - betha1) * -1 * delta_w_dictionary[i + 1][j][z]
                            vt[i + 1][j][z] = betha2 * vt[i + 1][j][z] + (1 - betha2) * (
                                    delta_w_dictionary[i + 1][j][z] ** 2)
                            mt_corrected[i + 1][j][z] = mt[i + 1][j][z] / (1 - betha1 ** t)
                            vt_corrected[i + 1][j][z] = vt[i + 1][j][z] / (1 - betha2 ** t)
                            weights[i][j][z] = weights[i][j][z] - alpha_adam * mt_corrected[i + 1][j][z] / (
                                    math.sqrt(vt_corrected[i + 1][j][z]) + e)

            last_error = error
            error = calculate_multi_layer_error(points, inner_layers, weights, nodes_count, output_nodes, b)

            if adaptative:
                if error < last_error:
                    if error_progress > 0:
                        error_progress = 0
                    error_progress -= 1
                if error > last_error:
                    if error_progress < 0:
                        error_progress = 0
                    error_progress += 1
                if error_progress == max_error_progress:
                    n -= n * 0.03
                if error_progress == -max_error_progress:
                    n += 0.01

            if error <= error_min:
                error_min = error
                min_weights = copy.deepcopy(weights)
        errors.append(error)
        stop_index += 1

    print(error_min)
    print(stop_index)

    return min_weights, errors, epocas


def init_weights(inner_layers: int, nodes_count: int, dim: int, output_nodes: int):
    weights = {}
    for j in range(inner_layers):
        weights[j] = []
        if j == 0:
            for i in range(nodes_count):
                weights[j].append(numpy.random.uniform(-1, 1, size=(dim + 1)))
        else:
            for i in range(nodes_count):
                weights[j].append(numpy.random.uniform(-1, 1, size=(nodes_count + 1)))

    weights[inner_layers] = []
    for i in range(output_nodes):
        weights[inner_layers].append(numpy.random.uniform(-1, 1, size=(nodes_count + 1)))

    return weights


def p_forward(inner_layers: int, weights: {}, nodes_count: int, output_nodes: int, point: Point, b: float):
    h_dictionary = {}
    o_dictionary = {0: point.e}  # [1, -1, 1]

    for i in range(inner_layers + 1):
        if i == inner_layers:
            h_dictionary[i + 1] = []
            o_dictionary[i + 1] = []
            for j in range(output_nodes):
                h_dictionary[i + 1].append(array_prod(o_dictionary[i], weights[i][j]))
                o_dictionary[i + 1].append(calculate_o(h_dictionary[i + 1][j], b))
        else:
            h_dictionary[i + 1] = []  # [1, o1, o2]
            o_dictionary[i + 1] = []
            o_dictionary[i + 1].append(1)
            for j in range(nodes_count):
                h_dictionary[i + 1].append(array_prod(o_dictionary[i], weights[i][j]))
                o_dictionary[i + 1].append(calculate_o(h_dictionary[i + 1][j], b))
    # el o_dictionary tiene las entradas de los nodos. Para la primer layer, tiene un punto que arranca con 1. Para las
    # demas, tiene la funcion de activacion( producto de las h * los weights) y un 1 appendeado al principio

    return h_dictionary, o_dictionary


def p_back(h_dictionary: {}, output_array: [], inner_layers: int, weights: {}, nodes_count: int, output_nodes: int,
           b: float, expected_value: float):
    error_dictionary = {}

    for i in reversed(range(inner_layers + 1)):
        if i == inner_layers:
            subtract_array = numpy.subtract(expected_value, output_array)
            error_dictionary[i + 1] = []
            for j in range(output_nodes):
                derivative = calculate_o_derivative(h_dictionary[i + 1][j], b)
                error_dictionary[i + 1].append(derivative * subtract_array[j])
        else:
            error_dictionary[i + 1] = []
            product_array = numpy.matmul(numpy.transpose(numpy.matrix(weights[i + 1])),
                                         numpy.transpose(numpy.matrix(error_dictionary[i + 2])))
            product_array_t = numpy.transpose(product_array)
            for j in range(nodes_count):
                derivative = calculate_o_derivative(h_dictionary[i + 1][j], b)
                error_dictionary[i + 1].append(derivative * numpy.ravel(product_array_t)[j + 1])
    return error_dictionary


def calculate_delta_w(o_dictionary: {}, error_dictionary: {}, n: float, inner_layers: int, nodes_count: int,
                      output_nodes: int, dim: int):
    delta_w_dictionary = {}
    for i in range(inner_layers + 1):
        delta_w_dictionary[i + 1] = []
        if i == 0:
            delta_w_dictionary[i + 1] = []
            for j in range(nodes_count):  # Por cada nodo calculo los nodes_count pesos
                delta_w_array = []
                for wi in range(dim + 1):
                    delta_w_array.append(n * error_dictionary[i + 1][j] * o_dictionary[i][wi])
                delta_w_dictionary[i + 1].append(delta_w_array)

        elif i == inner_layers:
            delta_w_dictionary[i + 1] = []
            for j in range(output_nodes):  # Por cada nodo calculo los nodes_count pesos
                delta_w_array = []
                for wi in range(nodes_count + 1):
                    delta_w_array.append(n * error_dictionary[i + 1][j] * o_dictionary[i][wi])
                delta_w_dictionary[i + 1].append(delta_w_array)
        else:
            delta_w_dictionary[i + 1] = []
            for j in range(nodes_count):  # Por cada nodo calculo los nodes_count pesos
                delta_w_array = []
                for wi in range(nodes_count + 1):
                    delta_w_array.append(n * error_dictionary[i + 1][j] * o_dictionary[i][wi])
                delta_w_dictionary[i + 1].append(delta_w_array)
    return delta_w_dictionary


def calculate_new_weights(delta_w_dictionary: {}, weights: {}, inner_layers: int, nodes_count: int, output_nodes: int):
    for i in range(inner_layers + 1):  # por cada layer

        count = nodes_count if i != inner_layers else output_nodes
        for j in range(count):  # por cada nodo en esa layer
            weights[i][j] += delta_w_dictionary[i + 1][j]
    return weights


def calculate_multi_layer_error(points: [], inner_layers: int, weights: {}, nodes_count: int, output_nodes: int,
                                b: float):
    total_error = 0
    for point in points:
        h_dictionary, o_dictionary = p_forward(inner_layers, weights, nodes_count, output_nodes, point, b)
        # TODO normalize
        # print(f" out: {o_dictionary[inner_layers + 1]}")
        # print(f" expected: {point.expected_value}")
        for i in range(output_nodes):
            total_error += (point.expected_value - o_dictionary[inner_layers + 1][i]) ** 2
        # print(f"{point.expected_value}  y el calculado {o_dictionary[inner_layers + 1][i]}")
    return total_error / len(points)


def calculate_o(h: float, b: float):
    return math.tanh(b * h)
    # return numpy.sign(h)


def calculate_o_derivative(h: float, b: float):
    return b * (1 - calculate_o(h, b) ** 2)
    # return 1


def array_prod(arr1: [], arr2: []):
    if len(arr1) != len(arr2):
        print(f"arr1: {arr1}")
        print(f"arr2: {arr2}")
        print("Array dim error")
        exit()
    arr_sum = 0
    for i in range(len(arr1)):
        arr_sum += (arr1[i] * arr2[i])
    return arr_sum


def calculate_excitement(point: declarations.Point, w: []):
    excitement = 0
    # print(point.e)
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
    wi = n * (point.normalized_expected_value - o) * b * (1 - o ** 2)
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


def calculate_error_non_linear(points: [], w: [], perceptron_type: str, b: float):
    error = 0
    for point in points:
        h = calculate_excitement(point, w)
        o = h
        if perceptron_type == 'non-linear-tan':
            o = math.tanh(h * b)
        elif perceptron_type == 'non-linear-logistic':
            o = 1 / (1 + math.e ** (-2 * b * h))
        # print(f"h: {h}, point: {point}")
        error += math.pow(point.normalized_expected_value - o, 2)

    return error * 1 / 2


def run(cot=1000, n=0.1, x=None, y=None, b=1, normalization='scale', perceptron_type='linear', inner_layers=2,
        nodes_count=2, momentum=False, adaptative=False, adam=False):
    config_file = open("config.json")
    config_data = json.load(config_file)
    if not config_data['config_by_code']:
        cot = config_data["cot"]
        n = config_data["n"]
        x = config_data["x"]
        y = config_data["y"]
        b = config_data["b"]
        inner_layers = config_data["inner_layers"]
        nodes_count = config_data["nodes_count"]
        normalization = config_data['normalization_type']
        perceptron_type = config_data["perceptron_type"]
        momentum = config_data["momentum"]
        adaptative = config_data["adaptative"]
        adam = config_data["adam"]


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
            sum_dev_squared += ((point.expected_value - mean) ** 2)
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
            if i < len(point_array) * 0.5:
                training_set.append(point_array[i])
            else:
                evaluation_set.append(point_array[i])
    elif perceptron_type == 'multi-layer-xor':
        for i in range(len(x)):
            arr = [1]
            for p in x[i]:
                arr.append(p)
            point = declarations.Point(arr, y[i])
            point_array.append(point)
            dim = len(x[0])
        if perceptron_type == 'non-linear-logistic':
            for point in point_array:
                point.expected_value = (point.expected_value - (-1)) / 2
        training_set = point_array
        evaluation_set = point_array
    elif perceptron_type == 'multi-layer-even':
        file = open('TP2-ej3-digitos.txt')
        lines = file.readlines()
        points = []
        number = [1]
        dim = 5 * 7
        for i in range(len(lines)):
            numbers = lines[i].split()
            for ni in numbers:
                number.append(int(ni))
            if (i + 1) % 7 == 0:
                expected = 1 if len(points) % 2 == 0 else -1  # Valor esperado un array
                points.append(declarations.Point(number, expected))
                number = [1]
        if perceptron_type == 'non-linear-logistic':
            for point in points:
                point.expected_value = (point.expected_value - (-1)) / 2
        for i in range(len(points)):
            if i < len(points) * 0.5:
                training_set.append(points[i])
            else:
                evaluation_set.append(points[i])

    elif perceptron_type == 'multi-layer-number':
        file = open('TP2-ej3-digitos.txt')
        lines = file.readlines()
        points = []
        number = [1]
        dim = 5 * 7
        for i in range(len(lines)):
            numbers = lines[i].split()
            for ni in numbers:
                number.append(int(ni))
            if (i + 1) % 7 == 0:
                expected = len(points)
                points.append(declarations.Point(number, expected))
                number = [1]

        for point in points:
            # if perceptron_type == 'non-linear-logistic':
            #    point.expected_value = point.expected_value / 9
            # if perceptron_type == 'non-linear-tan':
            point.expected_value = (point.expected_value * 2 / 9) - 1
        for i in range(len(points)):
            if i % 2 == 0:
                training_set.append(points[i])
            else:
                evaluation_set.append(points[i])

    if perceptron_type == 'multi-layer-xor' or perceptron_type == 'multi-layer-even' \
            or perceptron_type == 'multi-layer-number':
        perception, errors, epocas = multi_layer_perceptron_run(training_set, n, cot, dim, b, inner_layers, nodes_count,
                                                                1, momentum, adaptative, adam)
    else:
        perception = perceptron_run(training_set, n, cot, dim, perceptron_type, b)

    # print(perception)
    if perceptron_type == 'step':
        error = calculate_error_step(evaluation_set, perception.w)
    if perceptron_type == 'linear':
        error = calculate_error_linear(evaluation_set, perception.w)
    if perceptron_type == 'non-linear-tan' or perceptron_type == 'non-linear-logistic':
        error = calculate_error_non_linear(evaluation_set, perception.w, perceptron_type, b)
    if perceptron_type == 'multi-layer-xor' or perceptron_type == 'multi-layer-even' \
            or perceptron_type == 'multi-layer-number':
        error = calculate_multi_layer_error(evaluation_set, inner_layers, perception, nodes_count, 1, b)
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

    return perception, error, errors, epocas


if __name__ == "__main__":
    run()
