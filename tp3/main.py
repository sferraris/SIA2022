import tp2.main as funcs
import copy
import json
import math

import tp2.declarations
import random
import numpy
import matplotlib
import csv
from mpl_toolkits import mplot3d
from tp2.declarations import Point
from tp2.declarations import MultiPoint
import itertools
from scipy.optimize import minimize

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_font(value: bool):
    if value:
        return [
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]
        ]
    else:
        return []
def autoencoder_run(cot=1000, n=0.1, b=1, momentum=False, adaptative=False, adam=False, delta=0.2, percentage_train=0.5,
                    layers=None, powell=True, initial_weights=None, denoising_chance=0, custom_font = False):
    if layers is None:
        layers = []
    config_file = open("config.json")
    config_data = json.load(config_file)
    if not config_data['config_by_code']:
        cot = config_data["cot"]
        n = config_data["n"]
        b = config_data["b"]
        momentum = config_data["momentum"]
        adaptative = config_data["adaptative"]
        adam = config_data["adam"]
        delta = config_data["delta"]
        percentage_train = config_data["percentage_train"]
        layers = config_data["layers"]
        powell = config_data["powell"]
        denoising_chance = config_data["denoising_chance"]
        custom_font = config_data["custom_font"]

    if custom_font:
        font = get_font(True)
    else:
        font = [
            [1, 0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],
            [1, 0x00, 0x0e, 0x01, 0x0d, 0x13, 0x13, 0x0d],
            [1, 0x10, 0x10, 0x10, 0x1c, 0x12, 0x12, 0x1c],
            [1, 0x00, 0x00, 0x00, 0x0e, 0x10, 0x10, 0x0e],
            [1, 0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],
            [1, 0x00, 0x00, 0x0e, 0x11, 0x1f, 0x10, 0x0f],
            [1, 0x06, 0x09, 0x08, 0x1c, 0x08, 0x08, 0x08],
            [1, 0x0e, 0x11, 0x13, 0x0d, 0x01, 0x01, 0x0e],
            [1, 0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],
            [1, 0x00, 0x04, 0x00, 0x0c, 0x04, 0x04, 0x0e],
            [1, 0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0c],
            [1, 0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
            [1, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
            [1, 0x00, 0x00, 0x0a, 0x15, 0x15, 0x11, 0x11],
            [1, 0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
            [1, 0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e],
            [1, 0x00, 0x1c, 0x12, 0x12, 0x1c, 0x10, 0x10],
            [1, 0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],
            [1, 0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
            [1, 0x00, 0x00, 0x0f, 0x10, 0x0e, 0x01, 0x1e],
            [1, 0x08, 0x08, 0x1c, 0x08, 0x08, 0x09, 0x06],
            [1, 0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d],
            [1, 0x00, 0x00, 0x11, 0x11, 0x11, 0x0a, 0x04],
            [1, 0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a],
            [1, 0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11],
            [1, 0x00, 0x11, 0x11, 0x0f, 0x01, 0x11, 0x0e],
            [1, 0x00, 0x00, 0x1f, 0x02, 0x04, 0x08, 0x1f],
            [1, 0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],
            [1, 0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],
            [1, 0x0c, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0c],
            [1, 0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],
            [1, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f]
        ]

        font_binary = []
        for f in font:
            binary = [1]
            for i in range(len(f)):
                if i != 0:
                    binary += hex_to_binary(f[i])
            font_binary.append(binary)
        font = font_binary

    initial_points = []
    for i in range(len(font)):
        original_values = copy.deepcopy(font[i][1:])
        for j in range(len(font[i])):
            if j != 0:
                if font[i][j] == 1 and denoising_chance > 0:
                    r = random.randint(0, 1)
                    if r <= denoising_chance:
                        font[i][j] += random.randint(-1, 1) * delta
        initial_points.append(Point(font[i], original_values))


    # p_combinations = list(itertools.combinations(initial_points, 30))
    # print(len(p_combinations))
    # for l in p_combinations:
    #     weights, errors, epocas, accuracy_array = auto_encoder_run(l, n, cot, b, momentum, adaptative, adam,
    #                                                            layers)
    #     print(" ")


    weights, errors, epocas, accuracy_array, error_min = auto_encoder_run(initial_points, n, cot, b, momentum, adaptative,
                                                               adam, layers, powell, initial_weights)
    return weights, errors, epocas, accuracy_array, initial_points, error_min


def auto_encoder_run(points: [], n: float, cot: int, b: float, momentum: bool, adaptative: bool, adam: bool,
                     layers: [], powell: bool, initial_weights=None):
    stop_index = 0
    error_min = math.inf

    # ADAPTATIVE/MOMENTUM
    alpha = 0.8
    max_error_progress = 10
    # END ADAPTATIVE/MOMENTUM

    # ADAM
    alpha_adam = 0.001
    betha1 = 0.9
    betha2 = 0.999
    e = 10e-8
    t = 0
    mt = {}
    # ENDADAM

    # encoder_weights = funcs.init_weights(inner_layers, nodes_count, dim, output_nodes)
    # decoder_weights = funcs.init_weights(inner_layers, nodes_count, output_nodes, dim)

    if initial_weights is None:
        weights = init_weights(layers)
    else:
        weights = initial_weights

    # Parametros para salida, hay que redefinirlos pero no duplicarlos
    errors = []
    epocas = []
    accuracy_array = []
    # END paramnetros para salida

    # ADAM init vt y mt TODO si hacemos adam hay que hacerlo con los decoders tb
    # for i in range(len(encoder_weights)):
    #     mt[i + 1] = []
    #     for j in range(len(encoder_weights[i])):
    #         mt[i + 1].append([])
    #         for z in range(len(encoder_weights[i][j])):
    #             mt[i + 1][j].append(0)
    # vt = copy.deepcopy(mt)
    # mt_corrected = copy.deepcopy(mt)
    # vt_corrected = copy.deepcopy(mt)
    # END ADAM init weights

    min_weights = copy.deepcopy(weights)
    delta_w_dictionary_encoder = {}

    delta_w_dictionary_decoder = {}
    # Un solo error, TODO rehacer la funcion
    error = 0
    # END un solo error

    # Cree @micus que es para adaptative, vemos dsps
    error_progress = 0
    # END error progress

    while error_min > 1E-3 and stop_index < cot:
        print(f"{(stop_index / cot) * 100}% - {error} - {n}")
        indexes = list(range(len(points)))
        numpy.random.shuffle(indexes)
        # random_index = random.randint(0, len(points) - 1)
        epocas.append(stop_index)

        for random_index in indexes:
            t += 1
            point = points[random_index]
            # print("1")
            h_dictionary, o_dictionary = p_forward(layers, weights, point, b)
            # print("2")
            error_dictionary = p_back(h_dictionary, o_dictionary[len(layers) - 1], layers, weights, b,
                                      point.expected_value)
            if momentum and stop_index > 0:
                # print("3")
                previous_delta_w = copy.deepcopy(delta_w_dictionary)
                delta_w_dictionary = calculate_delta_w(o_dictionary, error_dictionary, n, layers)
                for i in range(len(delta_w_dictionary)):
                    for j in range(len(delta_w_dictionary[i + 1])):
                        for z in range(len(delta_w_dictionary[i + 1][j])):
                            delta_w_dictionary[i + 1][j][z] -= previous_delta_w[i + 1][j][z] * alpha
                # print("4")
                weights = calculate_new_weights(delta_w_dictionary, weights, layers)
            else:
                delta_w_dictionary = calculate_delta_w(o_dictionary, error_dictionary, n, layers)
                # print("4")
                weights = calculate_new_weights(delta_w_dictionary, weights, layers)

            # END adam, momentum

            last_error = error

            error = calculate_multi_layer_error(points, weights, layers, b)

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
                    n -= n * 0.0003
                if error_progress == -max_error_progress:
                    n += 0.0005

            if error <= error_min:
                error_min = error
                min_weights = weights
        errors.append(error)
        accuracy_array.append(accuracy_multi_layer(weights, layers, points, b))
        stop_index += 1

    print(f"epocas: {stop_index}")
    print(f"min training error: {error_min}")
    print(f"final accuracy: {accuracy_multi_layer(min_weights, layers, points, b)}")
    if powell:
        min_weights = minimize_weight(min_weights, points, b, layers)
        print(f"new error: {calculate_multi_layer_error(points, min_weights, layers, b)}")
        print(f"new accuracy: {accuracy_multi_layer(min_weights, layers, points, b)}")
    return min_weights, errors, epocas, accuracy_array, error_min


def init_weights(layers: []):
    weights = {}
    for j in range(len(layers)):
        if j != 0:
            weights[j] = []
            for i in range(layers[j]):
                weights[j].append(numpy.random.uniform(-1, 1, size=(layers[j - 1] + 1)))
    return weights


def p_forward(layers: [], weights: {}, encoder_initial_point: Point, b: float):
    h_dictionary = {}
    o_dictionary = {0: encoder_initial_point.e}  # [1, -1, 1]

    for i in range(len(layers)):
        if i != 0:
            h_dictionary[i] = []
            o_dictionary[i] = []
            if i != len(layers) - 1:
                o_dictionary[i].append(1)
            for j in range(layers[i]):
                previous_o = o_dictionary[i - 1]
                current_weights = weights[i][j]
                current_h = funcs.array_prod(previous_o, current_weights)
                current_o = funcs.calculate_o(current_h, b)
                h_dictionary[i].append(current_h)
                o_dictionary[i].append(current_o)
    return h_dictionary, o_dictionary


def p_back(h_dictionary: {}, output_array: [], layers: [], weights: {}, b: float, expected_value: []):
    error_dictionary = {}
    for i in reversed(range(len(layers))):
        if i != 0:
            if i == len(layers) - 1:
                subtract_array = numpy.subtract(expected_value, output_array)
                error_dictionary[i] = []
                for j in range(layers[i]):
                    current_h = h_dictionary[i][j]
                    derivative = funcs.calculate_o_derivative(current_h, b)
                    current_error = derivative * subtract_array[j]
                    error_dictionary[i].append(current_error)
            else:
                error_dictionary[i] = []
                current_weights = weights[i + 1]
                next_error = error_dictionary[i + 1]
                product_array = numpy.matmul(numpy.transpose(numpy.matrix(current_weights)),
                                             numpy.transpose(numpy.matrix(next_error)))
                product_array_t = numpy.transpose(product_array)
                for j in range(layers[i]):
                    current_h = h_dictionary[i][j]
                    derivative = funcs.calculate_o_derivative(current_h, b)
                    current_error = derivative * numpy.ravel(product_array_t)[j + 1]
                    error_dictionary[i].append(current_error)

    return error_dictionary


def calculate_delta_w(o_dictionary: {}, error_dictionary: {}, n: float, layers: []):
    delta_w_dictionary = {}
    for i in range(len(layers)):
        if i != 0:
            delta_w_dictionary[i] = []
            for j in range(layers[i]):  # Por cada nodo calculo los nodes_count pesos
                delta_w_array = []
                current_error = error_dictionary[i][j]
                for wi in range(layers[i - 1] + 1):
                    previous_o = o_dictionary[i - 1][wi]
                    delta_w_array.append(n * current_error * previous_o)
                delta_w_dictionary[i].append(delta_w_array)

    return delta_w_dictionary


def calculate_new_weights(delta_w_dictionary: {}, weights: {}, layers: []):
    for i in range(len(layers)):  # por cada layer
        if i != 0:
            for j in range(layers[i]):  # por cada nodo en esa layer
                weights[i][j] += delta_w_dictionary[i][j]
    return weights


def calculate_multi_layer_error(points: [], weights: {}, layers: [], b: float):
    total_error = 0
    dim = layers[0]
    for point in points:
        h_dictionary, o_dictionary = p_forward(layers, weights, point, b)
        for i in range(dim):
            total_error += (point.expected_value[i] - o_dictionary[len(layers) - 1][i]) ** 2
        # print(f"{point.expected_value}  y el calculado {o_dictionary[inner_layers + 1][i]}")
    return total_error / (len(points) * dim)


def calculate_multi_layer_error2(weights: [], layers: [], b: float, points: []):
    total_error = 0
    dim = layers[0]
    w = unflatten_weights(weights, layers)
    for point in points:
        h_dictionary, o_dictionary = p_forward(layers, w, point, b)
        for i in range(dim):
            total_error += (point.expected_value[i] - o_dictionary[len(layers) - 1][i]) ** 2
        # print(f"{point.expected_value}  y el calculado {o_dictionary[inner_layers + 1][i]}")
    # print(total_error / (len(points) * dim))
    return total_error / (len(points) * dim)


def accuracy_multi_layer(weights: {}, layers: {}, points: [], b: float):
    epsilon = 0.3
    win = 0
    dim = layers[0]
    final_layer = len(layers) - 1
    for point in points:
        h_dictionary, o_dictionary = p_forward(layers, weights, point, b)
        for i in range(dim):
            calculated_value = o_dictionary[final_layer][i]
            expected_value = point.expected_value[i]
            if calculated_value - epsilon < expected_value < calculated_value + epsilon:
                win += 1
        # print(f"{point.expected_value}  y el calculado {o_dictionary[inner_layers + 1][i]}")
    return win / (len(points) * dim)


def minimize_weight(weights: {}, points: [], b: float, layers: []):
    f_weights = flatten_weights(weights)
    minimized_weights = minimize(fun=calculate_multi_layer_error2, x0=f_weights, args=(layers, b, points),
                                 method="Powell", tol=0.01, options={'maxiter': 10})
    weights = unflatten_weights(minimized_weights.x, layers)
    return weights


def flatten_weights(weights: {}):
    f_weight = []
    for i in range(len(weights)):
        for w in weights[i + 1]:
            for elem in w:
                f_weight.append(elem)
    return numpy.array(f_weight)


def unflatten_weights(f_w, layers: []):
    weights = {}
    added = 0
    for i in range(len(layers)):
        if i != 0:
            weights[i] = []
            for j in range(layers[i]):
                weights[i].append([])
                for n in range(layers[i - 1] + 1):
                    weights[i][j].append(f_w[added])
                    added += 1
    return weights


def hex_to_binary(value):
    binary_array = [int(x) for x in bin(value)[2:].zfill(5)]
    return binary_array


if __name__ == "__main__":
    autoencoder_run()
