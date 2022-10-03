import numpy
from matplotlib import pyplot as plt

from main import run


def error_vs_n():
    learn = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    errors = []
    for n in learn:
        aux = []
        for i in range(10):
            perceptron, error = run(1000, n, None, None, 1, 'scale', 'non-linear-logistic')
            aux.append(error)
        errors.append(aux)
    y = []
    y_errors = []
    for errorN in errors:
        y.append(numpy.mean(errorN))
        y_errors.append(numpy.std(errorN))
    fig, ax = plt.subplots()
    #ax.set_yscale('log')
    ax.errorbar(learn, y,
                xerr=numpy.zeros(len(y_errors)),
                yerr=y_errors,
                fmt='-o')

    ax.set_xlabel('Tasa de Aprendizaje')
    ax.set_ylabel('Error')
    ax.set_title('Perceptron No Lineal')

    plt.show()



def main():
    error_vs_n()



if __name__ == "__main__":
    main()
