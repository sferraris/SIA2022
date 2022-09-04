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
        self.uses = 0

    def __str__(self):
        return f"r:{self.rgb[0]},g:{self.rgb[1]},b:{self.rgb[2]}, uses: {self.uses}, fitness: {self.fitness}"

    def __repr__(self):
        return f"r:{self.rgb[0]},g:{self.rgb[1]},b:{self.rgb[2]}, uses: {self.uses}, fitness: {self.fitness}"

    def __gt__(self, other):
        return self.fitness.__lt__(other.fitness)

    def __lt__(self, other):
        return self.fitness.__gt__(other.fitness)

    def __eq__(self, other):
        return self.rgb.__eq__(other.rgb)


def calculate_fitness(target_color: Color, color: Color):
    fitness_res = 255 * 3
    for pos in range(3):
        fitness_res -= abs(target_color.rgb[pos] - color.rgb[pos])
    color.fitness = fitness_res


def elite_selection_method(color_array: [], k: int):
    # receives a sorted_by_fitness color array
    selected_colors = []
    n = len(color_array)
    used_colors = 0
    for pos in range(len(color_array)):
        number_to_select = math.ceil((k - pos) / n)
        color = copy.deepcopy(color_array[pos])
        if used_colors + number_to_select <= k:
            color.uses = number_to_select
            used_colors += color.uses
            selected_colors.append(color)
        else:
            color.uses = k - used_colors
            used_colors += color.uses
            selected_colors.append(color)
            return selected_colors

    return selected_colors


def random_selection_method(color_array: [], k: int):
    used = []
    selected_colors = []
    if k >= len(color_array):
        return copy.deepcopy(color_array)

    for i in range(k):
        rand = random.randint(0, len(color_array) - 1)
        while used.__contains__(rand):
            rand = random.randint(0, len(color_array) - 1)
        used.append(rand)
        color = copy.deepcopy(color_array[rand])
        color.uses = 1
        selected_colors.append(color)

    return selected_colors


def deterministic_tournament_selection_method(color_array: [], k: int, m: int):
    if m > len(color_array):
        m = len(color_array)
    selected_colors = []

    for j in range(k):
        used = []
        aux_colors = []
        for i in range(m):
            rand = random.randint(0, len(color_array) - 1)
            while used.__contains__(rand) and selected_colors.__contains__(color_array[rand]):
                rand = random.randint(0, len(color_array) - 1)
            used.append(rand)
            color = copy.deepcopy(color_array[rand])
            color.uses = 1
            bisect.insort(aux_colors, color)
        selected_colors.append(aux_colors[0])

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
    # [c1, c2, c3, c4, c5]
    # [[rgb1, rgb2]]
    # [c1, c2, c3]

    used = []
    children_colors = []
    """
    for p1 in range(len(parent_array)):
        for p2 in range(len(parent_array)):
            first_parent = parent_array[p1]
            second_parent = parent_array[p2]
            if first_parent is not None \
                    and second_parent is not None \
                    and not first_parent.__eq__(second_parent) \
                    and not used.__contains__([p1, p2]) \
                    and not used.__contains__([p2, p1]):
                first_child, second_child = crossover(first_parent, second_parent)
                children_colors.append(first_child)
                children_colors.append(second_child)
                used.append([p1, p2])
                parent_array[p1] = None
                parent_array[p2] = None   
    """
    for p1 in range(len(parent_array)):
        first_parent = parent_array[p1]
        if first_parent.uses > 0:
            for p2 in range(p1 + 1, len(parent_array)):
                second_parent = parent_array[p2]
                if first_parent.uses > 0 and second_parent.uses > 0:
                    first_child, second_child = crossover(first_parent, second_parent)
                    first_parent.uses -= 1
                    second_parent.uses -= 1
                    children_colors.append(first_child)
                    children_colors.append(second_child)

    return children_colors


def print_color_palette(colors: []):
    print("Color Palette:")
    for color in colors:
        print(color)


def uniform_mutation(color: Color, mutation_probability, mutation_delta):
    color.rgb[0] = get_mutated_color(color.rgb[0], get_mutation_delta(mutation_probability, mutation_delta))
    color.rgb[1] = get_mutated_color(color.rgb[1], get_mutation_delta(mutation_probability, mutation_delta))
    color.rgb[2] = get_mutated_color(color.rgb[2], get_mutation_delta(mutation_probability, mutation_delta))


def gen_mutation(color: Color, mutation_probability, mutation_delta):
    rand = random.randint(0, 2)
    color.rgb[rand] = get_mutated_color(color.rgb[rand], get_mutation_delta(mutation_probability, mutation_delta))


def multi_gen_limited_mutation(color: Color, mutation_probability, mutation_delta):
    if is_mutation_delta(mutation_probability):
        quantity = random.randint(1, 3)
        start = random.randint(0, 2)
        for i in range(quantity):
            pos = (start + i) % 3
            color.rgb[pos] = get_mutated_color(color.rgb[pos], get_mutation_delta(101, mutation_delta))


def complete_mutation(color: Color, mutation_probability, mutation_delta):
    if is_mutation_delta(mutation_probability):
        color.rgb[0] = get_mutated_color(color.rgb[0], get_mutation_delta(101, mutation_delta))
        color.rgb[1] = get_mutated_color(color.rgb[1], get_mutation_delta(101, mutation_delta))
        color.rgb[2] = get_mutated_color(color.rgb[2], get_mutation_delta(101, mutation_delta))


def get_mutation_delta(mutation_probability, mutation_delta):
    mutation_decider = random.randint(0, 100)
    if mutation_decider < mutation_probability:
        return random.randint(-mutation_delta, mutation_delta)
    return 0


def get_mutated_color(color, delta):
    if color + delta < 0:
        return 0
    if color + delta > 255:
        return 255
    return color + delta


def is_mutation_delta(mutation_probability):
    mutation_decider = random.randint(0, 100)
    if mutation_decider < mutation_probability:
        return True
    return False


def run(target_color, mutation_probability, current_color_gen, mutation_delta, k, mutation, selection_method,
        max_cycles):
    count = 0
    is_mutation = (mutation_probability != 0 and mutation_delta != 0)
    while not current_color_gen[0].__eq__(target_color) \
            and (count < 1000 or is_mutation) \
            and (not is_mutation or (max_cycles != -1 and count < max_cycles and is_mutation)):
        count += 1
        # print(f"count: {count}, fitness: {current_color_gen[0].fitness}")
        # print_color_palette(current_color_gen)
        # Parent selection
        parent_colors = []
        if selection_method.__eq__("elite"):
            parent_colors = elite_selection_method(current_color_gen, k)
        if selection_method.__eq__("random"):
            parent_colors = random_selection_method(current_color_gen, k)
        if selection_method.__eq__("deterministic_tournament"):
            parent_colors = deterministic_tournament_selection_method(current_color_gen, k, 3)
        # Crossover
        children_colors = get_children(copy.deepcopy(parent_colors))
        # Child mutation
        for child in children_colors:
            if mutation.__eq__("uniform"):
                uniform_mutation(child, mutation_probability, mutation_delta)
            if mutation.__eq__("gen"):
                gen_mutation(child, mutation_probability, mutation_delta)
            if mutation.__eq__("multi_gen_limited"):
                multi_gen_limited_mutation(child, mutation_probability, mutation_delta)
            if mutation.__eq__("complete"):
                complete_mutation(child, mutation_probability, mutation_delta)

        # Fill-All implementation, cuando vuelve a entrar hace un elite selection
        current_color_gen = []
        for color in copy.deepcopy(parent_colors) + copy.deepcopy(children_colors):
            calculate_fitness(target_color, color)
            if not current_color_gen.__contains__(color):
                bisect.insort(current_color_gen, color)

    return current_color_gen, count


def get_random_population(population_length: int, target_color: Color):
    current_color_gen = []

    for color_data in range(population_length):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = Color(r, g, b)
        calculate_fitness(target_color, color)
        bisect.insort(current_color_gen, color)

    return current_color_gen


def run_statistics_mutation(population_length: int, target_color: Color, mutations: [], mutation_probability: int,
                            mutation_delta: int, k: int, selection_method: str, statistic_count: int, max_cycles: int):
    mutation_cycles = [0, 0, 0, 0]
    for j in range(statistic_count):
        print(j)
        current_color_gen = get_random_population(population_length, target_color)
        for i in range(len(mutations)):
            new_gen = copy.deepcopy(current_color_gen)
            new_gen, count = run(target_color, mutation_probability, new_gen, mutation_delta, k, mutations[i],
                                 selection_method, max_cycles)
            # print(f"Selected Color: {new_gen[0]}, fitness: {new_gen[0].fitness}, cycles: {count}")
            mutation_cycles[i] += count

    print(
        f"uniform: {mutation_cycles[0] / statistic_count}, gen: {mutation_cycles[1] / statistic_count}, "
        f"multi_gen_limited: {mutation_cycles[2] / statistic_count}, complete: {mutation_cycles[3] / statistic_count}")


def run_statistics_selection(population_length: int, target_color: Color, selection_methods: [],
                             mutation_probability: int,
                             mutation_delta: int, k: int, statistic_count: int, max_cycles: int, mutation_type: str):
    selection_cycles = [0, 0, 0]
    for j in range(statistic_count):
        print(j)
        current_color_gen = get_random_population(population_length, target_color)
        for i in range(len(selection_methods)):
            new_gen = copy.deepcopy(current_color_gen)
            new_gen, count = run(target_color, mutation_probability, new_gen, mutation_delta, k,
                                 mutation_type, selection_methods[i], max_cycles)
            selection_cycles[i] += count
    print(
        f"elite: {selection_cycles[0] / statistic_count}, random: {selection_cycles[1] / statistic_count}, "
        f"deterministic_tournament: {selection_cycles[2]/statistic_count}")


def main():
    config_file = open("config.json")
    config_data = json.load(config_file)
    mutations = ["uniform", "gen", "multi_gen_limited", "complete"]
    selection_methods = ["elite", "random", "deterministic_tournament"]

    try:
        target_color_data = config_data["target_color"]
        if len(target_color_data) != 3 or (not 0 <= target_color_data[0] < 256) or (
                not 0 <= target_color_data[1] < 256) or (not 0 <= target_color_data[2] < 256):
            print("Target color data must be a size 3 array of integers between 0 and 255")
            return
        population_length = config_data["population_length"]
        if population_length < 2:
            print("Population length must be bigger than 2")
            return
        k = config_data["K"]
        if k < 2:
            print("K must be bigger than 2")
            return
        selection_method = config_data["selection_method"]
        if not selection_methods.__contains__(selection_method):
            print(f"Selection method must be one of the following: {selection_methods}")
            return
        mutation_type = config_data["mutation_type"]
        if not mutations.__contains__(mutation_type):
            print(f"Mutation type must be one of the following: {mutations}")
            return
        mutation_probability = config_data["mutation_probability"]
        if not 0 <= mutation_probability <= 100:
            print("Mutation probability must be an integer between 0 and 100")
            return
        mutation_delta = config_data["mutation_delta"]
        if not 0 <= mutation_delta <= 25:
            print("Mutation delta must be an integer between 0 and 25")
            return
        max_cycles = config_data["max_cycles"]
        if max_cycles < -1:
            print("Max cycles must be an integer higher -1")
            return
        mutation_statistics = config_data["mutation_statistics"]
        selection_statistics = config_data["selection_statistics"]

    except:
        print("Invalid configuration, it should be something like this \n{\n\
          \"target_color\": [120, 230, 17],\n\
          \"population_length\": 20, \n\
          \"K\": 10, \n\
          \"mutation_probability\": 10,\n\
          \"mutation_delta\": 1, \n\
          \"mutation_type\": \"complete\",\n\
          \"selection_method\": \"deterministic_tournament\",\n\
          \"mutation_statistics\": true | false\n\
          \"selection_statistics\": true | false\n}")
        return

    target_color = Color(target_color_data[0], target_color_data[1], target_color_data[2])

    if mutation_statistics:
        run_statistics_mutation(population_length, target_color, mutations, mutation_probability,
                                mutation_delta, k, selection_method, 100, max_cycles)
    elif selection_statistics:
        run_statistics_selection(population_length, target_color, selection_methods, mutation_probability,
                                 mutation_delta, k, 100, max_cycles, mutation_type)

    else:
        current_color_gen = get_random_population(population_length, target_color)
        run(target_color, mutation_probability, current_color_gen, mutation_delta, k, mutation_type, selection_method,
            max_cycles)


if __name__ == "__main__":
    main()
