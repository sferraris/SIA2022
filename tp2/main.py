import copy
import json
import math

import declarations
import random
import numpy
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def simple_perceptron_step(points: [], n: float, cot: int, error: float, dim: int):
    i = 0
    w = []
    for j in range(dim + 1):
        w.append(0.0)
    error_min = math.inf
    perceptron: declarations.Perceptron = declarations.Perceptron(n, w)

    while error_min > 0 and i < cot:
        m = random.randint(0, len(points) - 1)
        h = calculate_excitement(points[m], perceptron.w)
        o = numpy.sign(h)
        delta_w = calculate_delta_w(points[m], perceptron.n, o)
        new_w: [] = calculate_w(delta_w, perceptron.w)
        error = calculate_error(points, new_w)
        if error <= error_min:
            error_min = error
            perceptron.w = new_w
        i += 1

    print(f"i: {i}")
    return perceptron


def calculate_excitement(point: declarations.Point, w: []):
    excitement = 0
    for i in range(len(w)):
        excitement += (point.e[i] * w[i])

    return excitement


def calculate_delta_w(point: declarations.Point, n: float, o: int):
    delta_w = []
    wi = n * (point.expected_value - o)
    for ei in point.e:
        delta_w.append(wi * ei)

    return delta_w


def calculate_w(delta_w: [], w: []):
    return numpy.sum([w, delta_w], axis=0)


def calculate_error(points: [], w: []):
    error = 0
    for point in points:
        h = calculate_excitement(point, w)
        o = numpy.sign(h)
        if o == 0:
            o = 1.0
        error += abs(point.expected_value - o)

    return error


def main():
    config_file = open("config.json")
    config_data = json.load(config_file)

    cot = config_data["cot"]
    error = config_data["error"]
    n = config_data["n"]
    x = config_data["x"]
    y = config_data["y"]

    point_array = []

    for i in range(len(x)):
        point = declarations.Point(x[i][0], x[i][1], y[i])
        point_array.append(point)

    perception = simple_perceptron_step(point_array, n, cot, error, len(x[0]))

    print(perception)

    for point in point_array:
        h = calculate_excitement(point, perception.w)
        o = numpy.sign(h)

    print(f"y = {perception.w[1] / (-1*perception.w[2])} * x + {perception.w[0] / (-1*perception.w[2])}")

    x = numpy.linspace(-5, 5, 100)
    y = perception.w[1] / (-1*perception.w[2]) * x + perception.w[0] / (-1*perception.w[2])
    plt.plot(x, y, '-r')

    for point in point_array:
        color = 'red'
        if point.expected_value == -1:
            color = 'black'
        plt.scatter(point.e[1], point.e[2], color=color)
    plt.show()


if __name__ == "__main__":
    main()
