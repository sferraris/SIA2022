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


class ColorMix:
    def __init__(self, proportions):
        self.proportions = proportions
        self.fitness = 0
        self.uses = 0

    def __gt__(self, other):
        return self.fitness.__lt__(other.fitness)

    def __lt__(self, other):
        return self.fitness.__gt__(other.fitness)

    def __eq__(self, other):
        return self.proportions.__eq__(other.proportions)

    def __str__(self):
        return f"Proportions: {self.proportions}, Fitness: {self.fitness}, Uses: {self.uses}"

    def __repr__(self):
        return f"Proportions: {self.proportions}, Fitness: {self.fitness}, Uses: {self.uses}"


def calculate_fitness(target_color: Color, color_mix: ColorMix, color_palette: []):
    fitness_res = 255 * 3
    rgb = get_rgb_from_mix(color_mix, color_palette)

    for pos in range(3):
        fitness_res -= abs(target_color.rgb[pos] - rgb[pos])
    color_mix.fitness = fitness_res


def get_rgb_from_mix(color_mix: ColorMix, color_palette: []):
    rgb = [0, 0, 0]
    for pos in range(len(color_palette)):
        rgb[0] += (color_mix.proportions[pos] / 100) * color_palette[pos].rgb[0]
        rgb[1] += (color_mix.proportions[pos] / 100) * color_palette[pos].rgb[1]
        rgb[2] += (color_mix.proportions[pos] / 100) * color_palette[pos].rgb[2]

    red = round(rgb[0])
    if red > 255:
        red = 255

    green = round(rgb[1])
    if green > 255:
        green = 255

    blue = round(rgb[2])
    if blue > 255:
        blue = 255

    return [red, green, blue]


def contains(color_mix_array: [], color_mix: ColorMix, color_palette):
    color_mix_rgb_array = []
    for color_mix_aux in color_mix_array:
        color_mix_rgb_array.append(get_rgb_from_mix(color_mix_aux, color_palette))
    color_mix_rgb = get_rgb_from_mix(color_mix, color_palette)

    return color_mix_rgb_array.__contains__(color_mix_rgb)


def normalize_proportions(proportions: []):
    res = []
    proportion_sum = 0
    for proportion in proportions:
        proportion_sum += proportion

    for i in range(len(proportions)):
        res_value = proportions[i] * 100 / proportion_sum
        res.append(res_value)

    return res


def elite_selection_method(color_mix_array: [], k: int):
    # receives a sorted_by_fitness color array
    selected_color_mix = []
    n = len(color_mix_array)
    used_colors = 0
    for pos in range(len(color_mix_array)):
        if used_colors < k:
            number_to_select = math.ceil((k - pos) / n)
            color = copy.deepcopy(color_mix_array[pos])
            if used_colors + number_to_select <= k:
                color.uses = number_to_select
                used_colors += color.uses
                selected_color_mix.append(color)
            else:
                color.uses = k - used_colors
                used_colors += color.uses
                selected_color_mix.append(color)

    return selected_color_mix


def random_selection_method(color_array: [], k: int, color_palette):
    used = []
    selected_colors = []
    if k >= len(color_array):
        for color_mix in color_array:
            color = copy.deepcopy(color_mix)
            color.uses = 1
            selected_colors.append(color)
        return selected_colors

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
    selected_positions = []

    if k >= len(color_array):
        for color_mix in color_array:
            color = copy.deepcopy(color_mix)
            color.uses = 1
            selected_colors.append(color)
        return selected_colors

    for j in range(k):
        used = []
        aux_colors = []
        for i in range(m):
            rand = random.randint(0, len(color_array) - 1)
            flag = False
            while (used.__contains__(rand) or selected_positions.__contains__(rand)) and not flag:
                if len(used) + len(selected_positions) == k:
                    flag = True
                rand = random.randint(0, len(color_array) - 1)
            if not flag:
                used.append(rand)
                color = copy.deepcopy(color_array[rand])
                color.uses = 1
                aux_colors.append(color)

        max_fitness = 0
        current_pos = -1
        for pos in range(len(aux_colors)):
            if aux_colors[pos].fitness > max_fitness:
                max_fitness = aux_colors[pos].fitness
                current_pos = pos

        selected_colors.append(aux_colors[current_pos])
        selected_positions.append(used[current_pos])

    return selected_colors


def crossover(c1: ColorMix, c2: ColorMix):
    first_child_proportions = []
    second_child_proportions = []
    first_child_proportion_sum = 0
    second_child_proportion_sum = 0
    for i in range(len(c1.proportions)):
        prob = random.randint(0, 1)
        if prob == 0:
            first_child_proportions.append(c1.proportions[i])
            first_child_proportion_sum += c1.proportions[i]
            second_child_proportions.append(c2.proportions[i])
            second_child_proportion_sum += c2.proportions[i]
        else:
            second_child_proportions.append(c1.proportions[i])
            second_child_proportion_sum += c1.proportions[i]
            first_child_proportions.append(c2.proportions[i])
            first_child_proportion_sum += c2.proportions[i]

    for i in range(len(c1.proportions)):
        first_child_proportion_value = first_child_proportions[i] * 100 / first_child_proportion_sum
        first_child_proportions[i] = first_child_proportion_value
        second_child_proportions_value = second_child_proportions[i] * 100 / second_child_proportion_sum
        second_child_proportions[i] = second_child_proportions_value

    first_child = ColorMix(first_child_proportions)
    second_child = ColorMix(second_child_proportions)

    return first_child, second_child


def get_children(parent_array: []):
    # [c1, c2, c3, c4, c5]
    # [[rgb1, rgb2]]
    # [c1, c2, c3]

    children_colors = []
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


def print_color_mix_array(colors: [], color_palette: []):
    print("Color Mix Array:")
    for color in colors:
        rgb = get_rgb_from_mix(color, color_palette)
        print(f"RGB: {rgb}, Fitness: {color.fitness}, Uses: {color.uses}")


def uniform_mutation(color_mix: ColorMix, mutation_probability, mutation_delta):
    for i in range(len(color_mix.proportions)):
        color_mix.proportions[i] = get_mutated_proportion(color_mix.proportions[i],
                                                          get_mutation_delta(mutation_probability, mutation_delta))


def gen_mutation(color_mix: ColorMix, mutation_probability, mutation_delta):
    rand = random.randint(0, len(color_mix.proportions) - 1)
    color_mix.proportions[rand] = get_mutated_proportion(color_mix.proportions[rand],
                                                         get_mutation_delta(mutation_probability, mutation_delta))


def multi_gen_limited_mutation(color_mix: ColorMix, mutation_probability, mutation_delta):
    if is_mutation_delta(mutation_probability):
        quantity = random.randint(1, len(color_mix.proportions))
        start = random.randint(0, len(color_mix.proportions) - 1)
        for i in range(quantity):
            pos = (start + i) % len(color_mix.proportions)
            color_mix.proportions[pos] = get_mutated_proportion(color_mix.proportions[pos],
                                                                get_mutation_delta(101, mutation_delta))


def complete_mutation(color_mix: ColorMix, mutation_probability, mutation_delta):
    if is_mutation_delta(mutation_probability):
        for i in range(len(color_mix.proportions)):
            color_mix.proportions[i] = get_mutated_proportion(color_mix.proportions[i],
                                                              get_mutation_delta(101, mutation_delta))


def get_mutation_delta(mutation_probability, mutation_delta):
    mutation_decider = random.randint(0, 100)
    if mutation_decider < mutation_probability:
        return random.randint(-mutation_delta, mutation_delta)
    return 0


def get_mutated_proportion(proportion, delta):
    if proportion + delta < 0:
        return 0
    if proportion + delta > 100:
        return 100
    return proportion + delta


def is_mutation_delta(mutation_probability):
    mutation_decider = random.randint(0, 100)
    if mutation_decider < mutation_probability:
        return True
    return False


def run(target_color, mutation_probability, current_color_gen, mutation_delta, k, mutation, selection_method,
        max_cycles, color_palette):
    count = 0
    is_mutation = (mutation_probability != 0 and mutation_delta != 0)
    current_rgb = get_rgb_from_mix(current_color_gen[0], color_palette)
    diff = 1
    while not current_rgb.__eq__(target_color.rgb) \
            and (count < 1000 or is_mutation) \
            and (not is_mutation or max_cycles == -1 or (count < max_cycles and is_mutation)) \
            :
        count += 1
        previous_color_gen = copy.deepcopy(current_color_gen)
        print(f"count: {count}, fitness: {current_color_gen[0].fitness}")
        # print_color_palette(current_color_gen)
        # Parent selection
        parent_colors = []
        if selection_method.__eq__("elite"):
            parent_colors = elite_selection_method(current_color_gen, k)
        if selection_method.__eq__("random"):
            parent_colors = random_selection_method(current_color_gen, k, color_palette)
        if selection_method.__eq__("deterministic_tournament"):
            parent_colors = deterministic_tournament_selection_method(current_color_gen, k, 3)

        print("Sale selection")
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

        for child_pos in range(len(children_colors)):
            children_colors[child_pos].proportions = normalize_proportions(children_colors[child_pos].proportions)

        # Fill-All implementation, cuando vuelve a entrar hace un elite selection
        current_color_gen = []
        for color in copy.deepcopy(children_colors) + copy.deepcopy(parent_colors):
            if color.fitness == 0:
                calculate_fitness(target_color, color, color_palette)
            if not contains(current_color_gen, color, color_palette):
                bisect.insort(current_color_gen, color)

        current_rgb = get_rgb_from_mix(current_color_gen[0], color_palette)

        diff = get_gen_difference(previous_color_gen, current_color_gen, color_palette)
        print(f"gen_diff: {len(current_color_gen)}")
        print(f"rgb: {current_rgb}")
        # print_color_mix_array(current_color_gen, color_palette)

    return current_color_gen, count


def get_gen_difference(gen1: [], gen2: [], color_palette):
    dif = 0
    length = min(len(gen1), len(gen2))
    for i in range(length):
        rgb1 = get_rgb_from_mix(gen1[i], color_palette)
        rgb2 = get_rgb_from_mix(gen2[i], color_palette)
        if not rgb1.__eq__(rgb2):
            dif += 1
    dif += abs(len(gen1) - len(gen2))
    return dif


def get_random_palette(population_length: int, target_color: Color):
    current_color_gen = []

    for color_data in range(population_length):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = Color(r, g, b)
        current_color_gen.append(color)

    return current_color_gen


def get_color_palette(color_palette: []):
    current_color_gen = []
    for i in range(len(color_palette)):
        r = color_palette[i][0]
        g = color_palette[i][1]
        b = color_palette[i][2]
        color = Color(r, g, b)
        current_color_gen.append(color)
    return current_color_gen


def get_random_population(population_length: int, target_color: Color, color_palette: []):
    current_color_mix_gen = []

    for color_mix_pos in range(population_length):
        proportions = []
        proportion_sum = 0
        for proportion in range(len(color_palette)):
            rand_proportion = random.randint(0, 100)
            proportion_sum += rand_proportion
            proportions.append(rand_proportion)
        for pos in range(len(proportions)):
            proportion_value = proportions[pos] * 100 / proportion_sum
            proportions[pos] = proportion_value
        color_mix = ColorMix(proportions)
        calculate_fitness(target_color, color_mix, color_palette)
        bisect.insort(current_color_mix_gen, color_mix)

    return current_color_mix_gen


def run_statistics_mutation(population_length: int, target_color: Color, mutations: [], mutation_probability: int,
                            mutation_delta: int, k: int, selection_method: str, statistic_count: int, max_cycles: int):
    mutation_cycles = [0, 0, 0, 0]
    for j in range(statistic_count):
        print(j)
        current_color_gen = get_random_palette(population_length, target_color)
        for i in range(len(mutations)):
            new_gen = copy.deepcopy(current_color_gen)
            new_gen, count = run(target_color, mutation_probability, new_gen, mutation_delta, k, mutations[i],
                                 selection_method, max_cycles)
            # print(f"Selected Color: {new_gen[0]}, fitness: {new_gen[0].fitness}, cycles: {count}")
            mutation_cycles[i] += count

    print(f"Selection method used: {selection_method}")
    print(f"Average cycles:")
    for i in range(len(mutations)):
        print(f"{mutations[i]}: {mutation_cycles[i] / statistic_count} cycles.")
    # print(
    #    f"uniform: {mutation_cycles[0] / statistic_count}, gen: {mutation_cycles[1] / statistic_count}, "
    #    f"multi_gen_limited: {mutation_cycles[2] / statistic_count}, complete: {mutation_cycles[3] / statistic_count}")


def run_statistics_selection(population_length: int, target_color: Color, selection_methods: [],
                             mutation_probability: int,
                             mutation_delta: int, k: int, statistic_count: int, max_cycles: int, mutation_type: str):
    selection_cycles = [0, 0, 0]
    for j in range(statistic_count):
        print(j)
        current_color_gen = get_random_palette(population_length, target_color)
        for i in range(len(selection_methods)):
            new_gen = copy.deepcopy(current_color_gen)
            new_gen, count = run(target_color, mutation_probability, new_gen, mutation_delta, k,
                                 mutation_type, selection_methods[i], max_cycles)
            selection_cycles[i] += count

    print(f"Mutation method used: {mutation_type}")
    print(f"Average cycles:")
    for i in range(len(selection_methods)):
        print(f"{selection_methods[i]}: {selection_cycles[i] / statistic_count} cycles.")

    # print(
    #    f"elite: {selection_cycles[0] / statistic_count}, random: {selection_cycles[1] / statistic_count}, "
    #    f"deterministic_tournament: {selection_cycles[2]/statistic_count}")


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
        if population_length < 3 and population_length != -1:
            print("Population length must be bigger than 3")
            return
        color_palette = config_data["color_palette"]
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

    color_palette_aux = get_color_palette(color_palette)

    if population_length != -1:
        color_palette_aux = get_random_palette(population_length, target_color)  # TODO remove param
    else:
        population_length = len(color_palette_aux)

    initial_population = get_random_population(population_length, target_color, color_palette_aux)

    current_color_gen, count = run(target_color, mutation_probability, initial_population, mutation_delta, k,
                                   mutation_type, selection_method,
                                   max_cycles, color_palette_aux)

    print(f"Finished in: {count}, color: {get_rgb_from_mix(current_color_gen[0], color_palette_aux)}")
    """
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
    """""


if __name__ == "__main__":
    main()
