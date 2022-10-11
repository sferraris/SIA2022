import math

import numpy
from matplotlib import pyplot as plt

from main import run

def error_vs_iter():
    perception, error, errors, iters, accuracy_train, accuracy_evaluation, accuracys = run(1000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                            'non-linear-logistic', 10, 10, False, False, False, False, 5, 0.1)
    fig, ax = plt.subplots()
    ax.errorbar(iters, errors,
                xerr=numpy.zeros(len(iters)),
                yerr=numpy.zeros(len(errors))
                )
    # ax.set_yscale('log')
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Error')
    ax.set_title('Error vs iteraciones')
    plt.show()

def accuracy_vs_iter():
    perception, error, errors, iters, accuracy_train, accuracy_evaluation, accuracys = run(1000, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                            'non-linear-logistic', 10, 10, False, False, False, False, 5, 0.1)
    fig, ax = plt.subplots()
    print(accuracys)
    ax.errorbar(iters, accuracys,
                xerr=numpy.zeros(len(iters)),
                yerr=numpy.zeros(len(errors))
                )
    # ax.set_yscale('log')
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Presicion')
    ax.set_title('Presicion vs iteraciones')
    plt.show()

def accuracy_vs_k():
    k_array = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    accuracy_arr = []

    for k in k_array:
        aux = []
        for i in range(10):
            perception, error, errors, iters, accuracy_train, accuracy_evaluation, accuracys = run(1000, 0.1,
                                                                                [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                [1, 1, -1, -1], 1, 'scale',
                                                                                'non-linear-tan', 10, 10, False,
                                                                                False, False, True, k, 0.1)
            aux.append(accuracy_evaluation)
        accuracy_arr.append(aux)

    y = []
    y_errors = []
    for errorN in accuracy_arr:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    ax.errorbar(k_array, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Cantidad de conjuntos')
    ax.set_ylabel('Precision')
    ax.set_title('Perceptron No Lineal')
    plt.show()


def accuracy_vs_n():
    n_array = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    accuracy_arr = []

    for n in n_array:
        aux = []
        for i in range(10):
            perception, error, errors, iters, accuracy_train, accuracy_evaluation, accuracys = run(1000, n,
                                                                                [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                [1, 1, -1, -1], 1, 'scale',
                                                                                'non-linear-tan', 10, 10, False,
                                                                                False, False, False, 5, 0.1)
            aux.append(accuracy_evaluation)
        accuracy_arr.append(aux)

    y = []
    y_errors = []
    for errorN in accuracy_arr:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    ax.errorbar(n_array, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Tasa de aprendizaje')
    ax.set_ylabel('Precision')
    ax.set_title('Perceptron No Lineal')
    plt.show()

def error_vs_n():
    n_array = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    error_arr = []

    for n in n_array:
        aux = []
        for i in range(10):
            perception, error, errors, iters, accuracy_train, accuracy_evaluation, accuracys = run(1000, n,
                                                                                [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                [1, 1, -1, -1], 1, 'scale',
                                                                                'non-linear-tan', 10, 10, False,
                                                                                False, False, False, 5, 0.1)
            aux.append(error)
        error_arr.append(aux)

    y = []
    y_errors = []
    for errorN in error_arr:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    ax.errorbar(n_array, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Tasa de aprendizaje')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron No Lineal')
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
    weight, error, errors, epocas, accuracy_train, accuracy_evaluation, accuracy_array = run(300, 0.1, [[-1, 1], [1, -1], [-1, -1], [1, 1]], [1, 1, -1, -1], 1, 'scale',
                                        'multi-layer-xor', 6, 6, False, False, True, True, 5, 0.2, 0.6)
    fig, ax = plt.subplots()
    ax.errorbar(epocas, errors,
                xerr=numpy.zeros(len(epocas)),
                yerr=numpy.zeros(len(errors))
                )

    ax.set_xlabel('Epocas')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron Multicapa')
    fig2, ax2 = plt.subplots()
    ax2.errorbar(epocas, accuracy_array,
                xerr=numpy.zeros(len(epocas)),
                yerr=numpy.zeros(len(accuracy_array))
                )

    ax2.set_xlabel('Epocas')
    ax2.set_ylabel('Precision')
    ax2.set_title('Perceptron Multicapa')

    plt.show()

def accuracy_vs_iters_multi():
    weight, error, errors, epocas, accuracy_train, accuracy_evaluation, accuracy_array = run(150, 0.1,
                                                                                             [[-1, 1], [1, -1],
                                                                                              [-1, -1], [1, 1]],
                                                                                             [1, 1, -1, -1], 1, 'scale',
                                                                                             'multi-layer-xor', 6, 6,
                                                                                             True, False, False, True,
                                                                                             5, 0.2, 0.6)
    fig, ax = plt.subplots()
    ax.errorbar(epocas, accuracy_array,
                xerr=numpy.zeros(len(epocas)),
                yerr=numpy.zeros(len(accuracy_array))
                )
    ax.set_xlabel('Epocas')
    ax.set_ylabel('Precision')
    ax.set_title('Perceptron Multicapa')
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

def error_vs_percentage_train():
    percentage = [ 0.1, 0.15,  0.2, 0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]
    errors = []
    for p in percentage:
        aux = []
        for i in range(10):
            weight, error, errores, iters, accuracy_train, accuracy_evaluation = run(1000, 0.1,
                                                                                 [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                 [1, 1, -1, -1], 1, 'scale',
                                                                                 'non-linear-tan', 10, 10, False, False,
                                                                                 False, False, 5, 0.1, p)
            aux.append(error)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(percentage, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Conjunto de Entrenamiento (%)')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron No Lineal')

    plt.show()

def error_vs_k():
    percentage = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    errors = []
    for p in percentage:
        aux = []
        for i in range(10):
            weight, error, errores, iters, accuracy_train, accuracy_evaluation = run(1000, 0.1,
                                                                                 [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                 [1, 1, -1, -1], 1, 'scale',
                                                                                 'non-linear-tan', 10, 10, False, False,
                                                                                 False, True, 5, 0.1, p)
            aux.append(error)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(percentage, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Cantidad de Conjuntos')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron No Lineal')

    plt.show()

def accuracy_vs_percentage_train():
    percentage = [ 0.1, 0.15,  0.2, 0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6, 0.65, 0.7,0.75, 0.8,0.85, 0.9, 0.95]
    errors = []
    for p in percentage:
        aux = []
        for i in range(10):
            weight, error, errores, iters, accuracy_train, accuracy_evaluation = run(1000, 0.1,
                                                                                 [[-1, 1], [1, -1], [-1, -1], [1, 1]],
                                                                                 [1, 1, -1, -1], 1, 'scale',
                                                                                 'non-linear-tan', 10, 10, False, False,
                                                                                 False, True, 5, 0.1, p)
            aux.append(accuracy_evaluation)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))

    fig, ax = plt.subplots()
    # ax.set_yscale('log')

    ax.errorbar(percentage, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Conjunto de Entrenamiento (%)')
    ax.set_ylabel('Precision')
    ax.set_title('Perceptron No Lineal')

    plt.show()
def main():
    # error_vs_n()
    # inner_layers()
    error_vs_iters_multi()
    #accuracy_vs_iters_multi()
    # error_vs_iter()
    # iter_vs_n()
<<<<<<< Updated upstream
    # compare_training()
    # error_vs_percentage_train()
    # error_vs_k()
    #accuracy_vs_percentage_train()
    error_vs_n()
=======
    accuracy_vs_n()
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
