import math

import numpy
from matplotlib import pyplot as plt

from main import run


def error_vs_n():
    learn = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    errors = []
    for n in learn:
        aux = []
        for i in range(10):
            perceptron, error, errores, iters = run(5000, n, None, None, 1, 'scale', 'linear')
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


def error_vs_iter():
    perceptron, error, errores, iters = run(1000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                            'step', 10, 10, False, False, False, False, 5, 0.1)
    fig, ax = plt.subplots()
    ax.errorbar(iters, errores,
                xerr=numpy.zeros(len(iters)),
                yerr=numpy.zeros(len(errores))
                )
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Error')
    ax.set_title('Error vs iteraciones')
    plt.show()


def iter_vs_n():
    # learn = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    learn = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    iterations = []
    for n in learn:
        aux = []
        print(n)
        print("")
        for i in range(20):
            perceptron, error, errores, iters = run(1000, n, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [-1, -1, -1, 1], 1,
                                                    'scale',
                                                    'step', 10, 10, False, False, False, False, 5, 0.1)
            aux.append(iters[-1])
        iterations.append(aux)
    y = []
    y_errors = []
    for errorN in iterations:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))
    minE = math.inf
    minN = 0
    # for i in range(len(learn)):
    #    print(f"{learn[i]}  {y[i]}")
    #    if y[i] < minE:
    #        minE = y[i]
    #        minN = learn[i]
    # print(f"{minN}  {minE}")
    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(learn, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Tasa de Aprendizaje')
    ax.set_ylabel('Iteraciones')
    ax.set_title('Perceptron Simple Escalon')

    plt.show()


def inner_layers():
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    nodes = 4
    errors = []
    for layer in layers:
        aux = []
        print(layer)
        for i in range(1):
            weight, error, errores, iters = run(5000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1,
                                                'scale',
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


def error_vs_iters_multi():
    weight, error, errors, epocas = run(300, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                        'multi-layer-even', 10, 10, True, False, False)
    fig, ax = plt.subplots()
    ax.errorbar(epocas, errors,
                xerr=numpy.zeros(len(epocas)),
                yerr=numpy.zeros(len(errors))
                )
    plt.show()


def compare_training():
    errors = []
    accuracy_train_array = []
    accuracy_eval_array = []
    iters_array = []
    for i in range(10):
        weight, error, errores, iters, accuracy_train, accuracy_evaluation = run(1000, 0.1,
                                                                                 [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                 [1, 1, -1, -1], 1, 'scale',
                                                                                 'non-linear-tan', 10, 10, False, False,
                                                                                 False, True, 5, 0.1)
        errors.append(error)
        accuracy_train_array.append(accuracy_train)
        accuracy_eval_array.append(accuracy_evaluation)
        iters_array.append(iters[-1] + 1)

    print("")
    print(f"Media de error: {numpy.mean(errors)}")
    print(f"Desviacion estandar de error: {numpy.std(errors)}")
    print(f"Media de accuracy entrenamiento: {numpy.mean(accuracy_train_array)}")
    print(f"Desviacion estandar entrenamiento: {numpy.std(accuracy_train_array)}")
    print(f"Media de accuracy evaluacion: {numpy.mean(accuracy_eval_array)}")
    print(f"Desviacion estandar evaluacion: {numpy.std(accuracy_eval_array)}")
    print(f"Media de cantidad iteraciones: {numpy.mean(iters_array)}")
    print(f"Desviacion estandar de cantidad iteraciones: {numpy.std(iters_array)}")


def main():
    # error_vs_n()
    # inner_layers()
    # error_vs_iters_multi()
    # error_vs_iter()
    # iter_vs_n()
    compare_training()


if __name__ == "__main__":
    main()
