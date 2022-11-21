import platform

import matplotlib
import numpy
from matplotlib import pyplot as plt
import tp2.main as funcs
from ast import literal_eval
from main import autoencoder_run
from main import p_forward
from tp2.declarations import Point

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


def graph_results_latent_space():
    inner_layers = 4
    nodes_count = 4
    output_nodes = 2
    b = 0.1
    n = 0.1
    cot = 100
    layers = [35, 20, 8, 2, 8, 20, 35]
    layers_aux = [35, 20, 8, 2]
    c = 0x60
    weights, errors, epocas, accuracy_array, initial_points = autoencoder_run(cot, n, b, False, False, False, 0.2, 0.5,
                                                                              layers, False)

    latent_points_x = []
    latent_points_y = []
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers_aux, weights, p,
                                                               b)
        # latent_points_x.append(o_dictionary_encoder[len(layers_aux)-1][0])
        # latent_points_y.append(o_dictionary_encoder[len(layers_aux) - 1][1])
        # latent_points.append(o_dictionary_encoder[])
        x = o_dictionary_encoder[len(layers_aux) - 1][0]
        y = o_dictionary_encoder[len(layers_aux) - 1][1]
        plt.scatter([x], [y])
        plt.text(x, y, chr(c))
        c += 1
    print("hola2")
    # plt.scatter(latent_points_x, latent_points_y)
    plt.show()


def show_letters():
    b = 0.8
    n = 0.01
    cot = 5000
    layers = [35, 28, 15, 7, 2, 7, 15, 28, 35]
    weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, True, False, 0.2, 0.5,
                                                                              layers, False)

    # for p in initial_points:
    #     h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p, b)
    #     font_char = p.e[1:]
    #     s = ""
    #     for char in range(35):
    #         if char % 5 == 0 and char != 0:
    #             print(s)
    #             s = ""
    #         c = 'X' if font_char[char] > 0 else '.'
    #         s = s + c
    #     print(s)
    #     print()
    #     font_char = o_dictionary_encoder[len(layers) - 1]
    #     s = ""
    #     for char in range(35):
    #         if char % 5 == 0 and char != 0:
    #             print(s)
    #             s = ""
    #         c = 'X' if font_char[char] > 0 else '.'
    #         s = s + c
    #     print(s)
    #     print()

    value = 1
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p, b)
        font_char = p.e[1:]
        a = []
        v = []
        # plt.subplot(2, 35, value)
        plt.subplot(1, 35, value)
        for char in range(35):
            if char % 5 == 0 and char != 0:
                a.append(v)
                v = []

            v.append(font_char[char])
        a.append(v)
        array = numpy.array(a)
        bounds = [-6, -2, 2, 6]
        plt.imshow(1 - array, interpolation='nearest',
                         cmap="gray")
        plt.axis('off')
        value += 1

    plt.savefig("./Results/Img/res_input_adaptative.png", bbox_inches='tight', dpi=1800)
    value = 1
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p, b)
        plt.subplot(1, 35, value)
        font_char = o_dictionary_encoder[len(layers) - 1]
        a = []
        v = []
        for char in range(35):
            if char % 5 == 0 and char != 0:
                a.append(v)
                v = []

            # v.append(1 if font_char[char] > 0.5 else -1)
            v.append(font_char[char])
        a.append(v)
        array = numpy.array(a)
        img = plt.imshow(1 - array, interpolation='nearest',
                         cmap="gray")
        plt.axis('off')

        value += 1

        plt.savefig("./Results/Img/res_output_adaptative.png", bbox_inches='tight', dpi=1800)


def error_vs_epocas():
    b = 0.8
    n = 0.01
    cot = 1000
    layers_0 = [35, 20, 8, 2, 8, 20, 35]
    layers_1 = [35, 30, 25, 20, 15, 10, 5, 2, 5, 10, 15, 20, 25, 30, 35]
    layers_2 = [35, 25, 15, 5, 2, 5, 15, 25, 35]
    layers_3 = [35, 24, 13, 2, 13, 24, 35]
    layers_4 = [35, 27, 19, 11, 2, 11, 19, 27, 35]
    layers_5 = [35, 28, 15, 7, 2, 7, 15, 28, 35]
    layers_6 = [35, 19, 11, 7, 5, 2, 5, 7, 11, 19, 35]
    layers_7 = [35, 2, 35]
    layers_8 = [35, 20, 10, 5, 2, 5, 10, 20, 35]
    layers_9 = []
    for i in reversed(range(36)):
        if i > 1:
            layers_9.append(i)
    for i in range(36):
        if i > 2:
            layers_9.append(i)
    layers_10 = [35, 31, 23, 19, 17, 13, 11, 7, 5, 3, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 35]
    all_layers = [layers_0, layers_1, layers_2]
    c = 0x61
    all_labels = [0, 1, 2]

    for j in range(len(all_layers)):
        weights, errors, epocas, accuracy_array, initial_points = autoencoder_run(cot, n, b, False, False, False, 0.2,
                                                                                  0.5,
                                                                                  all_layers[j], False)
        #save_autoencoder(weights,all_layers[j],all_labels[j])
        f = open(f"./Results/ErrorVsEpocas/layer_{all_labels[j]}", "w")
        for i in range(len(epocas)):
            f.write(f"{epocas[i]} {errors[i]}\n")
        f.close()
        c += 1

def save_autoencoder(weights: {}, layers: [], layer_id):
    f = open(f"./Results/Autoencoders/layer_{layer_id}", 'w')
    f.write(str(layers) + "\n")
    for j in range(len(layers)):
        if j != 0:
            for i in range(layers[j]):
                f.write("[")
                for w in weights[j][i]:
                    f.write(str(w) + ", ")
                f.write("]\n")
    f.close()

def read_autoencoder(layer_id: int):
    f = open(f"./Results/Autoencoders/layer_{layer_id}")
    lines = f.readlines()
    layers = literal_eval(lines[0])
    weights = {}
    index = 1
    weights[index] = []
    for i in range(len(lines[1:])):
        weight = literal_eval(lines[1:][i])
        weights[index].append(weight)
        if len(weights[index]) == layers[index]:
            index += 1
            if index != len(layers):
                weights[index] = []
    f.close()
    return weights, layers
def error_vs_epocas_graph():
    fig, ax = plt.subplots()
    c = 0x61
    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for label in all_labels:
        epocas = []
        errors = []
        f = open(f"./Results/ErrorVsEpocas/layer_{label}")
        lines = f.readlines()
        for line in lines:
            info_arr = line.split(' ')
            epocas.append(int(info_arr[0]))
            errors.append(float(info_arr[1].split('\n')[0]))
        f.close()
        ax.plot(epocas, errors, label=chr(c), linewidth=1)
        c += 1

    ax.set_xlabel('Epocas')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')
    ax.legend()
    fig.savefig("./Results/ErrorVsEpocas/ErorVsEpocasArquis.png", bbox_inches='tight', dpi=1800)

    #plt.show()
def error_vs_b():
    fig, ax = plt.subplots()
    b_values = [0.83]
    n = 0.01
    cot = 300
    layers_5 = [35, 28, 15, 7, 2, 7, 15, 28, 35]
    x_vals = []
    y_errors = []
    y_vals = []
    for b in b_values:
        min_errors = []
        f = open(f"./Results/error_vs_b/b_res{b}", "w")
        for i in range(5):
            print(f"{b} {i}")
            weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False, False, 0.2,
                                                                                      0.5,
                                                                                      layers_5, False)
            min_errors.append(error_min)
            f.write(str(error_min) + "\n")
        f.close()
        y_vals.append(numpy.mean(min_errors))
        y_errors.append(numpy.std(min_errors))
        x_vals.append(b)
    ax.errorbar(x_vals, y_vals, xerr=numpy.zeros(len(y_vals)), yerr=y_errors, fmt='-o')
    ax.set_xlabel('b')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')

    fig.savefig("./Results/error_vs_b/result.png", bbox_inches='tight', dpi=1800)

def graph_error_vs_b():
    fig, ax = plt.subplots()
    x_vals = []
    y_errors = []
    y_vals = []
    b_values = [0.73, 0.75, 0.78, 0.83, 0.85, 0.87, 0.93, 0.95]
    b_values2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    b_values = b_values + b_values2
    b_values.sort()
    for i in range(len(b_values)):
        f = open(f"./Results/error_vs_b/b_res{b_values[i]}")
        lines = f.readlines()
        min_errors = []
        for line in lines:
            info_arr = line.split(' ')
            min_errors.append(float(info_arr[0]))
        y_vals.append(numpy.mean(min_errors))
        y_errors.append(numpy.std(min_errors))
        x_vals.append(b_values[i])
        f.close()
    ax.errorbar(x_vals, y_vals, xerr=numpy.zeros(len(y_vals)), yerr=y_errors, fmt='-o')
    ax.set_xlabel('b')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')

    fig.savefig("./Results/error_vs_b/result.png", bbox_inches='tight')

    #plt.show()
def n_and_epochs_combined():
    b = 0.8
    layers = [35, 28, 15, 7, 2, 7, 15, 28, 35]
    c = 0x61
    all_labels = ['0.1_1000', '0.1_3000', '0.1_5000', '0.01_1000', '0.01_3000', '0.01_5000', '0.001_1000', '0.001_3000',
                  '0.001_5000']

    n_array = [0.1, 0.01, 0.001]
    cot_array = [1000, 3000, 5000]
    label_index = 0
    for n_index in range(len(n_array)):
        for cot_index in range(len(cot_array)):
            weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot_array[cot_index], n_array[n_index], b, False, False, False, 0.2,
                                                                                      0.5,
                                                                                      layers, False)
            save_autoencoder(weights, layers, all_labels[label_index])
            f = open(f"./Results/NAndEpochs/layer_{all_labels[label_index]}", "w")
            for i in range(len(epocas)):
                f.write(f"{epocas[i]} {errors[i]}\n")
            f.close()
            label_index += 1
            c += 1
def error_vs_epocas_graph_n_epochs():
    fig, ax = plt.subplots()
    all_labels = ['0.1_1000', '0.1_3000', '0.1_5000', '0.01_1000', '0.01_3000', '0.01_5000', '0.001_1000', '0.001_3000',
                  '0.001_5000']

    n_array = [0.1, 0.01, 0.001]
    cot_array = [1000, 3000, 5000]
    for label in all_labels:
        epocas = []
        errors = []
        f = open(f"./Results/NAndEpochs/layer_{label}")
        lines = f.readlines()
        for line in lines:
            info_arr = line.split(' ')
            epocas.append(int(info_arr[0]))
            errors.append(float(info_arr[1].split('\n')[0]))
        f.close()
        ax.plot(epocas, errors, label=label, linewidth=0.75)

    ax.set_xlabel('Epocas')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')
    ax.legend()
    fig.savefig("./Results/NAndEpochs/ErorVsEpocas.png", bbox_inches='tight', dpi=1800)

def main():
    show_letters()
    #error_vs_epocas()
    #error_vs_epocas_graph()
    # save_autoencoder({}, [0, 1, 2, 3], 84)
    #n_and_epochs_combined()
    #error_vs_b()
    #graph_error_vs_b()
    #error_vs_epocas_graph_n_epochs()


if __name__ == "__main__":
    main()
