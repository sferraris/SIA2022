import copy
import platform

import matplotlib
import numpy
from PIL import Image
from matplotlib import pyplot as plt
import tp2.main as funcs
from ast import literal_eval
from main import autoencoder_run
from main import p_forward
from main import get_font
from tp2.declarations import Point

if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


def graph_results_latent_space():
    b = 0.8
    n = 0.01
    cot = 5000
    layers = [35, 28, 15, 7, 2, 7, 15, 28, 35]
    layers_aux = [35, 28, 15, 7, 2]
    c = 0x60
    weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False, False,
                                                                                         0.2, 0.5,
                                                                                         layers, True)
    save_autoencoder(weights, layers, 'denoising-l5')

    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers_aux, weights, p,
                                                               b)

        x = o_dictionary_encoder[len(layers_aux) - 1][0]
        y = o_dictionary_encoder[len(layers_aux) - 1][1]
        plt.scatter([x], [y])
        plt.text(x, y, chr(c))
        c += 1
    # plt.scatter(latent_points_x, latent_points_y)
    # plt.show()
    plt.savefig("./Results/Img/latent-space.png", bbox_inches='tight', dpi=1800)


def show_letters():
    b = 0.8
    n = 0.01
    cot = 5000
    layers = [35, 27, 19, 11, 2, 11, 19, 27, 35]
    weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False, False,
                                                                                         0.8, 0.5,
                                                                                         layers, False, None, 0.2)
    save_autoencoder(weights, layers, "denoising_0.8_powell")


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

    plt.savefig("./Results/Img/res_input_denoising_0.8_powell.png", bbox_inches='tight', dpi=1800)
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

    plt.savefig("./Results/Img/res_output_denoising_0.8_powell.png", bbox_inches='tight', dpi=1800)


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
    all_layers = [layers_4, layers_5]
    c = 0x61
    all_labels = ['denoising_0.2', 'denoising_0.2']

    for j in range(len(all_layers)):
        weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False,
                                                                                             False, 0.2,
                                                                                             0.5,
                                                                                             all_layers[j], False, None,
                                                                                             0.2)
        save_autoencoder(weights, all_layers[j], all_labels[j])
        f = open(f"./Results/ErrorVsEpocasDenoising/layer_{all_labels[j]}", "w")
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


def read_autoencoder(layer_id):
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
    # all_labels = ['denoising_0.2_4', 'denoising_0.2_5']
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
    # fig.savefig("./Results/ErrorVsEpocas/ErorVsEpocasArquis.png", bbox_inches='tight', dpi=1800)

    plt.show()


def error_vs_denoising():
    fig, ax = plt.subplots()
    b = 0.8
    denoising_values = [0.8, 0.9, 1]
    n = 0.01
    cot = 300
    layers_4 = [35, 27, 19, 11, 2, 11, 19, 27, 35]
    x_vals = []
    y_errors = []
    y_vals = []
    for denoising in denoising_values:
        min_errors = []
        f = open(f"./Results/ErrorVsDenoising/denoising_{denoising}", "w")
        for i in range(5):
            print(f"{denoising} {i}")
            weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False,
                                                                                                 False, False, 0.2,
                                                                                                 0.5,
                                                                                                 layers_4, False, None,
                                                                                                 denoising)
            min_errors.append(error_min)
            f.write(str(error_min) + "\n")
        f.close()
        y_vals.append(numpy.mean(min_errors))
        y_errors.append(numpy.std(min_errors))
        x_vals.append(denoising)
    ax.errorbar(x_vals, y_vals, xerr=numpy.zeros(len(y_vals)), yerr=y_errors, fmt='-o')
    ax.set_xlabel('Denoising Chance')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')

    fig.savefig("./Results/ErrorVsDenoising/result.png", bbox_inches='tight', dpi=1800)


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
            weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False,
                                                                                                 False, False, 0.2,
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

    # plt.show()


def graph_error_vs_denoising():
    fig, ax = plt.subplots()
    x_vals = []
    y_errors = []
    y_vals = []
    denoising_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in range(len(denoising_values)):
        f = open(f"./Results/ErrorVsDenoising/denoising_{denoising_values[i]}")
        lines = f.readlines()
        min_errors = []
        for line in lines:
            info_arr = line.split(' ')
            min_errors.append(float(info_arr[0]))
        y_vals.append(numpy.mean(min_errors))
        y_errors.append(numpy.std(min_errors))
        x_vals.append(denoising_values[i])
        f.close()
    ax.errorbar(x_vals, y_vals, xerr=numpy.zeros(len(y_vals)), yerr=y_errors, fmt='-o')
    ax.set_xlabel('Denoising Chance')
    ax.set_ylabel('Error')
    ax.set_title('Autoencoder')

    fig.savefig("./Results/ErrorVsDenoising/result.png", bbox_inches='tight', dpi=1800)

    plt.show()


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
            weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot_array[cot_index],
                                                                                                 n_array[n_index], b,
                                                                                                 False, False, False,
                                                                                                 0.2,
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


def latent_space_letter_generator():
    weights, layers = read_autoencoder("south_park_500_3_2_3")
    b = 0.8
    size = len(layers)
    print(layers)
    print(layers[(int(size / 2)):])
    autoencoder_layers = layers[(int(size / 2)):]
    print(int(size / 2) + 1)
    new_w = {}
    letter_size = 9
    row_size = 3
    for i in range(int(size / 2)):
        new_w[i + 1] = weights[i + int(size / 2) + 1]

    x_values = []
    for i in range(21):
        x_values.append(-1 + i * 0.1)
    y_values = copy.deepcopy(x_values)
    points = []
    fig, axs = plt.subplots(len(x_values), len(y_values), figsize=(12, 12))
    count1: int = 0
    count2: int = 0
    count = 0
    for x in x_values:
        for y in reversed(y_values):
            p = Point([1, x, y], [])
            h_dictionary_encoder, o_dictionary_encoder = p_forward(autoencoder_layers, new_w, p, 0.8)
            a = []
            v = []
            font_char = o_dictionary_encoder[int(size / 2)]
            for char in range(letter_size):
                if char % row_size == 0 and char != 0:
                    a.append(v)
                    v = []

                # v.append(1 if font_char[char] > 0.5 else -1)
                v.append(font_char[char])
            a.append(v)
            array = numpy.array(a)
            ax = axs[count1, count2]
            ax.imshow(1 - array, interpolation='nearest', cmap="gray")
            ax.axis('off')
            #ax.set_title(f'[{round(x, 2)}, {round(y, 2)}]', fontsize=3)
            count1 += 1
        count2 += 1
        count1 = 0
    fig.tight_layout(pad=0.5)

    plt.savefig(f"./Results/NewLetters/letter_gen_big_custom.png", bbox_inches='tight', dpi=1800)


def latent_space_letter_generator_single():
    weights, layers = read_autoencoder("denoising-l5")
    b = 0.8
    size = len(layers)
    print(layers)
    print(layers[(int(size / 2)):])
    autoencoder_layers = layers[(int(size / 2)):]
    print(int(size / 2) + 1)
    new_w = {}
    for i in range(int(size / 2)):
        new_w[i + 1] = weights[i + int(size / 2) + 1]

    x_values = [1, -0.4, -0.1, -1, 0, -0.4]
    y_values = [1, -0.1, -0.6, -1, -0.8, 0.9]
    points = []
    count = 0
    for i in range(len(x_values)):
        print(i)
        x = x_values[i]
        y = y_values[i]
        p = Point([1, x, y], [])
        h_dictionary_encoder, o_dictionary_encoder = p_forward(autoencoder_layers, new_w, p, 0.8)
        a = []
        v = []
        font_char = o_dictionary_encoder[int(size / 2)]
        for char in range(35):
            if char % 5 == 0 and char != 0:
                a.append(v)
                v = []
            v.append(font_char[char])
        a.append(v)
        array = numpy.array(a)
        plt.imshow(1 - array, interpolation='nearest', cmap="gray")
        plt.axis('off')
        plt.title(f'({x}, {y})')
        count += 1
        plt.savefig(f"./Results/NewLetters/letter_gen_{x}_{y}.png", bbox_inches='tight', dpi=1800)


def show_letters_custom():
    b = 0.8
    n = 0.01
    cot = 3000
    layers = [9, 6, 3, 2, 3, 6, 9]
    font_size = 8
    letter_size = 9
    row_size = 3

    weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False, False,
                                                                                         0.2, 0.5,
                                                                                         layers, True, None, 0, True)

    save_autoencoder(weights, layers, 'show_letters_custom_9_6_3_2_powell')
    value = 1
    for p in initial_points:
        font_char = p.e[1:]
        a = []
        v = []
        plt.subplot(1, font_size, value)
        for char in range(letter_size):
            if char % row_size == 0 and char != 0:
                a.append(v)
                v = []

            v.append(font_char[char])
        a.append(v)
        array = numpy.array(a)
        plt.imshow(1 - array, interpolation='nearest',
                   cmap="gray")
        plt.axis('off')
        value += 1

    plt.savefig(f"./Results/CustomImg/res_input.png", bbox_inches='tight', dpi=1800)
    value = 1
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p, b)
        plt.subplot(1, font_size, value)
        font_char = o_dictionary_encoder[len(layers) - 1]
        a = []
        v = []
        for char in range(letter_size):
            if char % row_size == 0 and char != 0:
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

        plt.savefig("./Results/CustomImg/res_output.png", bbox_inches='tight', dpi=1800)

def show_letters_custom_images():
    w, h = 18, 18


    b = 0.8
    n = 0.01
    cot = 1
    layers1 = [324, 212, 106, 53, 26, 3, 2, 3, 26, 53, 106, 212, 324]
    layers2 = [324, 10, 3, 2, 3, 10, 324]
    layers3 = [324, 200, 100, 50, 2, 50, 100, 200, 324]
    layers4 = [324, 150, 50, 2, 50, 150, 324]
    layers5 = [324, 300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50, 25, 2, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 324]
    layersArray = [ layers5, layers4, layers3, layers2, layers1]
    all_labels = [ "layers5", "layers4", "layers3", "layers2", "layers1"]
    font_size = 4
    letter_size = 30000

    print("Autoencoder")
    for i in range(len(layersArray)):
        weights, errors, epocas, accuracy_array, initial_points, error_min = autoencoder_run(cot, n, b, False, False, False,
                                                                                        0.2, 0.5,
                                                                                     layersArray[i], False, None, 0, True)
        save_autoencoder(weights, layersArray[i], f"numbers_{all_labels[i]}")
        f = open(f"./Results/Numbers/ErrorVsEpocas/layer_{all_labels[i]}", "w")
        for i in range(len(epocas)):
            f.write(f"{epocas[i]} {errors[i]}\n")
        f.close()
    """
    index = 1
    for p in initial_points:
        h_dictionary_encoder, o_dictionary_encoder = p_forward(layers, weights, p, b)
        font_char = o_dictionary_encoder[len(layers) - 1]
        max = numpy.max(font_char)
        min = numpy.min(font_char)
        image = []
        for pixel in font_char:
            image.append(int(255*((pixel - min)/(max-min))))
        data = numpy.reshape(image, (w, h))
        img = Image.fromarray(data.astype('uint8'), 'L')
        img.save(f'./Results/Numbers/gen2_{index}.png')
        index += 1
"""
def latent_space_letter_custom_gen():
    weights, layers = read_autoencoder("numbers")
    b = 0.8
    size = len(layers)
    autoencoder_layers = layers[(int(size / 2)):]
    new_w = {}

    for i in range(int(size / 2)):
        new_w[i + 1] = weights[i + int(size / 2) + 1]

    x_values = []
    for i in range(11):
        x_values.append(-1 + i * 0.2)
    y_values = copy.deepcopy(x_values)
    points = []
    count = 0
    w, h = 28, 28
    for x in x_values:
        for y in reversed(y_values):
            p = Point([1, x, y], [])
            h_dictionary_encoder, o_dictionary_encoder = p_forward(autoencoder_layers, new_w, p, 0.8)
            font_char = o_dictionary_encoder[len(autoencoder_layers) - 1]
            image = []
            max = numpy.max(font_char)
            min = numpy.min(font_char)
            for pixel in font_char:
                image.append(int(255 * ((pixel - min) / (max - min))))
            data = numpy.reshape(image, (w, h, 4))
            img = Image.fromarray(data.astype('uint8'), 'RGBA')
            img.save(f'./Results/Numbers/LatentSpace/gen_{round(x, 2)}_{round(y, 2)}.png')
            count += 1
def main():
    # graph_results_latent_space()
    # show_letters()
    # error_vs_epocas()
    # error_vs_epocas_graph()
    # save_autoencoder({}, [0, 1, 2, 3], 84)
    # n_and_epochs_combined()
    # error_vs_b()
    # graph_error_vs_b()
    # error_vs_epocas_graph_n_epochs()
    # error_vs_denoising()
    # graph_error_vs_denoising()
    # latent_space_letter_generator_single()
    # latent_space_letter_generator()
    # graph_error_vs_denoising()
    # show_letters_custom()
    show_letters_custom_images()
    # latent_space_letter_custom_gen()


if __name__ == "__main__":
    main()
