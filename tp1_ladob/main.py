import copy
import json
import math
import bisect
from enum import Enum
import random


class Color:
    def __init__(self, red, green, blue):
        self.rgb = [red, green, blue]
        self.fitness = 0

    def __str__(self):
        return f"r:{self.rgb[0]},g:{self.rgb[1]},b:{self.rgb[2]}"

    def __repr__(self):
        return f"r:{self.rgb[0]},g:{self.rgb[1]},b:{self.rgb[2]}"

    def __gt__(self, other):
        return self.fitness.__gt__(other.fitness)

    def __lt__(self, other):
        return self.fitness.__lt__(other.fitness)

    def __eq__(self, other):
        return self.rgb.__eq__(other.rgb)


def calculate_fitness(target_color: Color, color: Color):
    fitness_res = 255 * 3
    for pos in range(3):
        fitness_res -= abs(target_color.rgb[pos] - color.rgb[pos])
    color.fitness = fitness_res


def elite_selection_mode(color_array: [], k: int):
    # receives a sorted_by_fitness color array
    selected_colors = []
    n = len(color_array)
    for pos in range(len(color_array)):
        number_to_select = math.ceil((k - pos) / n)
        if number_to_select > 0:
            for p in range(number_to_select):
                if len(selected_colors) < k:
                    selected_colors.append(copy.deepcopy(color_array[pos]))

    return selected_colors


def crossover(c1: Color, c2: Color):
    first_child_rgb = []
    second_child_rgb = []
    for i in range(3):
        prob = random.randint(0, 1)
        if prob == 0:
            first_child_rgb.append(c1.rgb[i])
            second_child_rgb.append(c2.rgb[i])
        else:
            second_child_rgb.append(c1.rgb[i])
            first_child_rgb.append(c2.rgb[i])
    first_child = Color(first_child_rgb[0], first_child_rgb[1], first_child_rgb[2])
    second_child = Color(second_child_rgb[0], second_child_rgb[1], second_child_rgb[2])

    return first_child, second_child


def get_children(parent_array: []):
    # [c1., c1, c2., c2, c3]
    # [[rgb1, rgb2]]
    # [c1, c2, c3]
    #
    used = []
    children_colors = []
    print(parent_array)
    for p1 in range(len(parent_array)):
        for p2 in range(len(parent_array)):
            first_parent = parent_array[p1]
            second_parent = parent_array[p2]
            if first_parent is not None \
                    and second_parent is not None \
                    and not first_parent.__eq__(second_parent) \
                    and not used.__contains__([p1, p2]) \
                    and not used.__contains__([p2, p1]):
                print("entra")
                first_child, second_child = crossover(first_parent, second_parent)
                children_colors.append(first_child)
                children_colors.append(second_child)
                used.append([p1, p2])
                parent_array[p1] = None
                parent_array[p2] = None

    print(used)
    return children_colors


def print_color_palette(colors: []):
    print("Color Palette:")
    for color in colors:
        print(color)


def main():
    config_file = open("config.json")
    config_data = json.load(config_file)

    target_color_data = config_data["target_color"]
    population_length = config_data["population_length"]
    K = config_data["K"]

    target_color = Color(target_color_data[0], target_color_data[1], target_color_data[2])
    current_color_gen = []

    for color_data in range(population_length):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = Color(r, g, b)
        calculate_fitness(target_color, color)
        bisect.insort(current_color_gen, color)

    print_color_palette(current_color_gen)
    max_fitness = 0
    # Para cada i, hay que hacer selection, crossover, mutation
    for i in range(10):
        parent_colors = elite_selection_mode(current_color_gen, K)
        children_colors = get_children(copy.deepcopy(parent_colors))
        # TODO mutation ver que onda dsps, preguntarle a PAU
        # Fill-All implementation, cuando vuelve a entrar hace un elite selection
        current_color_gen = []
        for color in copy.deepcopy(parent_colors) + copy.deepcopy(children_colors):
            calculate_fitness(target_color, color)
            if color.fitness > max_fitness:
                max_fitness = color.fitness
            bisect.insort(current_color_gen, color)

    print_color_palette(current_color_gen)

    print(
        f"Selected Color: {current_color_gen[0]}, fitness: {current_color_gen[0].fitness}, max_fitness: {max_fitness}")


if __name__ == "__main__":
    main()