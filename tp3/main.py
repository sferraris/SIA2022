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

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def autoencoder_run():
    # run(cot=1000, n=0.1, x=None, y=None, b=1, normalization='scale', perceptron_type='linear', inner_layers=2,
    #         nodes_count=2, momentum=False, adaptative=False, adam=False, cross_validation=False, k=1, delta=0.2,
    #         percentage_train = 0.5):

    # x e y -> siempre None
    # normalizacion siempre scale
    # perceptron_type siempre multilayer-nuevoej (crear nuevo en codigo)
    # inner layers, nodes count, a definir segun convenga
    # adam siempre False
    # momentum, adaptative, a convenir
    # cross validation siempre false porque lo hicimos mal, k no importa
    # delta es para el noise
    # percentage train probablemente sea siempre 1, pero a definir por las preguntas del tp

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
    initial_points = []
    for i in range(len(font)):
        initial_points.append(Point(font[i], font[i]))

    min_weights_encoder, errors, epocas, accuracy_array = auto_encoder_run(initial_points, 0.1, 500, 7, 0.1, 2, 2, 2, False, False, False)

    print("Hola Micus")


def auto_encoder_run(points: [], n: float, cot: int, dim: int, b: float, inner_layers: int, nodes_count: int,
                     output_nodes: int, momentum: bool, adaptative: bool, adam: bool):
    stop_index = 0
    error_min = math.inf

    # ADAPTATIVE/MOMENTUM
    alpha = 0.8
    max_error_progress = 2
    # END ADAPTATIVE/MOMENTUM

    # ADAM
    alpha_adam = 0.001
    betha1 = 0.9
    betha2 = 0.999
    e = 10e-8
    t = 0
    mt = {}
    # ENDADAM

    encoder_weights = funcs.init_weights(inner_layers, nodes_count, dim, output_nodes)
    decoder_weights = funcs.init_weights(inner_layers, nodes_count, output_nodes, dim)

    # Parametros para salida, hay que redefinirlos pero no duplicarlos
    errors = []
    epocas = []
    accuracy_array = []
    # END paramnetros para salida

    # ADAM init vt y mt TODO si hacemos adam hay que hacerlo con los decoders tb
    for i in range(len(encoder_weights)):
        mt[i + 1] = []
        for j in range(len(encoder_weights[i])):
            mt[i + 1].append([])
            for z in range(len(encoder_weights[i][j])):
                mt[i + 1][j].append(0)
    vt = copy.deepcopy(mt)
    mt_corrected = copy.deepcopy(mt)
    vt_corrected = copy.deepcopy(mt)
    # END ADAM init weights

    min_weights_encoder = copy.deepcopy(encoder_weights)
    delta_w_dictionary_encoder = {}

    min_weights_decoder = copy.deepcopy(encoder_weights)
    delta_w_dictionary_decoder = {}
    # Un solo error, TODO rehacer la funcion
    error = 0
    # END un solo error

    # Cree @micus que es para adaptative, vemos dsps
    error_progress = 0
    # END error progress

    while error_min > 1E-3 and stop_index < cot:
        print(f"{(stop_index/cot) * 100}%")
        indexes = list(range(len(points)))
        numpy.random.shuffle(indexes)
        # random_index = random.randint(0, len(points) - 1)
        epocas.append(stop_index)

        for random_index in indexes:
            t += 1
            encoder_initial_point = points[random_index]

            h_dictionary_encoder, o_dictionary_encoder = funcs.p_forward(inner_layers, encoder_weights, nodes_count,
                                                                         output_nodes, encoder_initial_point, b)

            initial_arr = [1]
            for i in range(len(o_dictionary_encoder[inner_layers + 1])):
                initial_arr.append(o_dictionary_encoder[inner_layers + 1][i])

            decoder_initial_point = Point(initial_arr, encoder_initial_point.e)

            h_dictionary_decoder, o_dictionary_decoder = funcs.p_forward(inner_layers, decoder_weights, nodes_count,
                                                                         dim, decoder_initial_point, b)

            error_dictionary = p_back(h_dictionary_encoder, h_dictionary_decoder,
                                      o_dictionary_decoder[inner_layers + 1], inner_layers, encoder_weights,
                                      decoder_weights, nodes_count, output_nodes, dim, b,
                                      encoder_initial_point.e[1:])

            # ACA antes estaba adam, momentum
            error_dictionary_decoder = dict((k, v) for (k, v) in error_dictionary.items() if k >= inner_layers + 1)
            for i in range(inner_layers + 1):
                error_dictionary_decoder[i + 1] = error_dictionary_decoder[i + 1 + inner_layers + 1]



            delta_w_dictionary_decoder = funcs.calculate_delta_w(o_dictionary_decoder, error_dictionary_decoder, n,
                                                                 inner_layers, nodes_count, dim, output_nodes)


            delta_w_dictionary_encoder = funcs.calculate_delta_w(o_dictionary_encoder, error_dictionary, n,
                                                                 inner_layers, nodes_count, output_nodes, dim)

            decoder_weights = funcs.calculate_new_weights(delta_w_dictionary_decoder, decoder_weights, inner_layers,
                                                          nodes_count, dim)

            encoder_weights = funcs.calculate_new_weights(delta_w_dictionary_encoder, encoder_weights, inner_layers,
                                                          nodes_count, output_nodes)
            # END adam, momentum

            last_error = error
            error = calculate_multi_layer_error(points, inner_layers, encoder_weights, decoder_weights, nodes_count,
                                                output_nodes, dim, b)

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
                min_weights_encoder = copy.deepcopy(encoder_weights)
        errors.append(error)
        accuracy_array.append(
            funcs.accuracy_multi_layer(encoder_weights, points, inner_layers, nodes_count, output_nodes, b))
        stop_index += 1

    print(f"epocas: {stop_index}")
    print(f"min training error: {error_min}, min weights encoder: {min_weights_encoder}")

    return min_weights_encoder, errors, epocas, accuracy_array


def p_back(h_dictionary_encoder: {}, h_dictionary_decoder: {}, output_array_decoder: [], inner_layers: int,
           weights_encoder: {}, weights_decoder: {}, nodes_count: int, output_nodes: int, dim: int, b: float,
           expected_value: []):
    error_dictionary = {}

    for i in reversed(range(inner_layers + 1)):  # 3 + 1 -> 0...3 / -> 3 * 2 + 2 = 8 / 3...0 + 4
        if i == inner_layers:
            subtract_array = numpy.subtract(expected_value, output_array_decoder)
            error_dictionary[inner_layers + 1 + i + 1] = []
            for j in range(dim):
                derivative = funcs.calculate_o_derivative(h_dictionary_decoder[i + 1][j], b)
                error_dictionary[inner_layers + 1 + i + 1].append(derivative * subtract_array[j])
        else:
            error_dictionary[inner_layers + 1 + i + 1] = []
            product_array = numpy.matmul(numpy.transpose(numpy.matrix(weights_decoder[i + 1])),
                                         numpy.transpose(numpy.matrix(error_dictionary[inner_layers + 1 + i + 2])))
            product_array_t = numpy.transpose(product_array)
            for j in range(nodes_count):
                derivative = funcs.calculate_o_derivative(h_dictionary_decoder[i + 1][j], b)
                error_dictionary[inner_layers + 1 + i + 1].append(derivative * numpy.ravel(product_array_t)[j + 1])

    # parte del decoder
    # MAPEO
    # Seguir errores
    for i in reversed(range(inner_layers + 1)):
        if i == inner_layers:
            error_dictionary[i + 1] = []
            product_array = numpy.matmul(numpy.transpose(numpy.matrix(weights_decoder[0])),
                                         numpy.transpose(numpy.matrix(error_dictionary[i + 2])))
            product_array_t = numpy.transpose(product_array)
            for j in range(output_nodes):
                derivative = funcs.calculate_o_derivative(h_dictionary_decoder[1][j], b)
                error_dictionary[i + 1].append(derivative * numpy.ravel(product_array_t)[j + 1])
        else:
            error_dictionary[i + 1] = []
            product_array = numpy.matmul(numpy.transpose(numpy.matrix(weights_encoder[i + 1])),
                                         numpy.transpose(numpy.matrix(error_dictionary[i + 2])))
            product_array_t = numpy.transpose(product_array)
            for j in range(nodes_count):
                derivative = funcs.calculate_o_derivative(h_dictionary_encoder[i + 1][j], b)
                error_dictionary[i + 1].append(derivative * numpy.ravel(product_array_t)[j + 1])

    return error_dictionary


def calculate_multi_layer_error(points: [], inner_layers: int, encoder_weights: {}, decoder_weights: {},
                                nodes_count: int, output_nodes: int, dim: int, b: float):
    total_error = 0
    for point in points:
        h_dictionary_encoder, o_dictionary_encoder = funcs.p_forward(inner_layers, encoder_weights, nodes_count,
                                                                     output_nodes, point, b)

        initial_arr = [1]
        for i in range(len(o_dictionary_encoder[inner_layers + 1])):
            initial_arr.append(o_dictionary_encoder[inner_layers + 1][i])

        decoder_initial_point = Point(initial_arr, point.e)

        h_dictionary_decoder, o_dictionary_decoder = funcs.p_forward(inner_layers, decoder_weights, nodes_count,
                                                                     dim, decoder_initial_point, b)

        for i in range(dim):
            total_error += (point.e[i] - o_dictionary_decoder[inner_layers + 1][i]) ** 2
        # print(f"{point.expected_value}  y el calculado {o_dictionary[inner_layers + 1][i]}")
    return total_error / (len(points) * dim)


if __name__ == "__main__":
    autoencoder_run()
