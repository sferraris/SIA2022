import math

import numpy
from matplotlib import pyplot as plt

from main import run


def error_vs_n():
    learn = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    errors = []
    for n in learn:
        aux = []
        for i in range(10):
            perceptron, error = run(5000, n, None, None, 1, 'scale', 'linear')
            aux.append(error)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))
    minE = math.inf
    minN = 0
    for i in range(len(learn)):
        print(f"{learn[i]}  {y[i]}")
        if y[i] < minE:
            minE = y[i]
            minN = learn[i]
    print(f"{minN}  {minE}")
    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(learn, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Tasa de Aprendizaje')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron Lineal')

    plt.show()


def inner_layers():
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    nodes = 4
    errors = []
    for layer in layers:
        aux = []
        print(layer)
        for i in range(1):
            weight, error = run(5000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                'multi-layer-xor', layer, nodes)
            aux.append(error)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))
    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(layers, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Inner layers')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron Multicapa')

    plt.show()


def main():
    # error_vs_n()
    #inner_layers()
    weight, error, errors, epocas = run(1000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                        'multi-layer-even', 5, 5, False, False, False)
    fig, ax = plt.subplots()
    ax.errorbar(epocas, errors,
                xerr=numpy.zeros(len(epocas)),
                yerr=numpy.zeros(len(errors))
                )
    plt.show()


if __name__ == "__main__":
    main()
