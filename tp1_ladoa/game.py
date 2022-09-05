import operator
import time
import bisect

from grid import MyGame
import arcade
import copy
import json
import sys

# Set how many rows and columns we will have
HEADER_COUNT = 2

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 32

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

# Do the math to figure out our screen dimensions
SCREEN_TITLE = "Game"


class State:
    def __init__(self, color, colored_cells, border_cells, grid, moves, cells_to_paint, heuristic, cost):
        self.color = color
        self.cells_to_paint = cells_to_paint
        self.colored_cells = colored_cells
        self.border_cells = border_cells
        self.grid = grid
        self.moves = moves
        self.win = 0
        self.heuristic = heuristic
        self.cost = cost

    def __hash__(self):
        return hash(str(self.grid))

    def __str__(self) -> str:
        return f"{get_color(self.color)}, {self.__hash__()}"

    def __repr__(self) -> str:
        return f"{get_color(self.color)}, {self.__hash__()}"


class Node:
    def __init__(self, state: State):
        self.state = state
        self.children = []
        self.parent = None

    def __str__(self) -> str:
        # return "{" + f" color: {get_color(self.color)},moves: {self.moves},  win: {self.win}, total_cells: {len(self.border_cells) + len(self.colored_cells)}, children: {self.children} " + "}"
        return f"{get_color(self.state.color)} -> {self.children} "

    def __repr__(self) -> str:
        return f"{get_color(self.state.color)} -> {self.children} "

    def __getitem__(self, key):
        return getattr(self, key)

    def __gt__(self, other):
        if self.state.heuristic + self.state.cost == other.state.heuristic + other.state.cost:
            return self.state.heuristic.__gt__(other.state.heuristic)
        return (self.state.heuristic + self.state.cost).__gt__(other.state.heuristic + other.state.cost)

    def __lt__(self, other):
        if self.state.heuristic + self.state.cost == other.state.heuristic + other.state.cost:
            return self.state.heuristic.__lt__(other.state.heuristic)
        return (self.state.heuristic + self.state.cost).__lt__(other.state.heuristic + other.state.cost)


def get_child(node: Node, color: int, game: MyGame, total_moves: int, matrix_size: int, heuristic):
    child_color = color
    cells_to_paint = game.get_related_cells(copy.deepcopy(node.state.border_cells), color, node.state.grid,
                                            copy.deepcopy(node.state.border_cells))
    child_cells_to_paint = len(cells_to_paint)
    child_grid = copy.deepcopy(node.state.grid)
    child_moves = node.state.moves + 1

    painted = copy.deepcopy(node.state.colored_cells) + copy.deepcopy(node.state.border_cells)
    for cell in painted:
        r = cell[0]
        c = cell[1]
        child_grid[r][c] = color

    child_colored_cells = copy.deepcopy(node.state.colored_cells)
    child_border_cells = copy.deepcopy(node.state.border_cells) + cells_to_paint

    for cell in child_border_cells:
        if not game.is_border(cell[0], cell[1], child_grid):
            child_border_cells.remove(cell)
            child_colored_cells.append(cell)

    child_heuristic = 0
    child_cost = 0

    if heuristic is not None:
        if heuristic == "cells_left_cost":
            child_heuristic = matrix_size * matrix_size - (len(child_border_cells) + len(child_colored_cells))
            child_cost = child_moves
        if heuristic == "cells_left":
            child_heuristic = matrix_size * matrix_size - (len(child_border_cells) + len(child_colored_cells))
        if heuristic == "colors_left":
            child_heuristic = game.calculate_remaining_colors(copy.deepcopy(child_colored_cells),
                                                              copy.deepcopy(child_border_cells),
                                                              copy.deepcopy(child_grid))
        if heuristic == "colors_left_cost":
            child_heuristic = game.calculate_remaining_colors(copy.deepcopy(child_colored_cells),
                                                              copy.deepcopy(child_border_cells),
                                                              copy.deepcopy(child_grid))
            child_cost = child_moves
        if heuristic == "shortest_path":
            aux = State(child_color, child_colored_cells, child_border_cells, child_grid, 0,
                        0, 0, 0)
            child_heuristic = shortest_path(game, aux, matrix_size, copy.deepcopy(game.color_count))
        if heuristic == "shortest_path_cost":
            aux = State(child_color, child_colored_cells, child_border_cells, child_grid, 0,
                        0, 0, 0)
            child_heuristic = shortest_path(game, aux, matrix_size, copy.deepcopy(game.color_count))
            child_cost = child_moves
    child_state = State(child_color, child_colored_cells, child_border_cells, child_grid, child_moves,
                        child_cells_to_paint, child_heuristic, child_cost)
    child = Node(child_state)
    return copy.deepcopy(child)


expanded = 0
cant_frontier = 0


def DFS(game: MyGame, color_number: int, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    tree = Node(tree_state)
    visited = set()
    global expanded
    expanded += 1
    DFS_rec(tree, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))
    return tree if tree.state.win else None


def DFS_rec(node: Node, game: MyGame, color_number: int, total_moves: int, matrix_size: int, visited: set):
    visited.add(node.state.__hash__())
    global expanded
    global cant_frontier
    for color in range(color_number):
        if color != node.state.color and node.state.win == 0:
            child = get_child(node, color, game, total_moves, matrix_size, None)
            if not visited.__contains__(child.state.__hash__()):
                if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                    child.state.win = 1
                    node.state.win = 1
                    visited.add(child.state)
                    node.children.append(child)
                    cant_frontier += 1
                    return 1
                elif total_moves == -1 or child.state.moves < total_moves:
                    expanded += 1
                    return_value = DFS_rec(child, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))
                    if return_value == 1:
                        node.state.win = 1
                        node.children.append(child)
                        return 1
                else:
                    cant_frontier += 1


def BFS(game: MyGame, color_number: int, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    return BFS_alg(tree, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))


def BFS_alg(tree: Node, game: MyGame, color_number: int, total_moves: int, matrix_size: int, visited: set):
    bfs_queue = [tree]
    global expanded
    global cant_frontier
    while bfs_queue:
        node = bfs_queue.pop(0)
        expanded += 1
        visited.add(node.state.__hash__())
        node.state.win = 1
        if total_moves == -1 or node.state.moves < total_moves:
            children = []
            analyzed = 0
            for color in range(color_number):
                if color != node.state.color:
                    child = get_child(node, color, game, total_moves, matrix_size, None)
                    if not visited.__contains__(child.state.__hash__()):
                        child.parent = node
                        node.children.append(child)
                        children.append(child)
                        analyzed += 1
                        if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                            cant_frontier += len(bfs_queue) + analyzed
                            return get_solution(child)
            bfs_queue += children

    return None


def local_greedy(game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    global expanded
    expanded += 1
    local_greedy_rec(tree, game, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))
    return tree if tree.state.win else None


def local_greedy_rec(node: Node, game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int,
                     visited: set):
    visited.add(node.state.__hash__())
    children_array = []
    global expanded
    global cant_frontier
    for color in range(color_number):
        if color != node.state.color:
            child = get_child(node, color, game, total_moves, matrix_size, heuristic)
            if not visited.__contains__(child.state.__hash__()):
                bisect.insort(children_array, child)
    analyzed = 0
    for child in children_array:
        analyzed += 1
        vis = []
        if node.state.win == 0:
            vis.append(child)
            if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                child.state.win = 1
                node.state.win = 1
                node.children.append(child)
                cant_frontier += analyzed
                return 1
            elif total_moves == -1 or child.state.moves < total_moves:
                expanded += 1
                cant_frontier += len(children_array) - analyzed
                return_value = local_greedy_rec(child, game, color_number, heuristic, total_moves, matrix_size,
                                                copy.deepcopy(visited))
                if return_value == 1:
                    node.state.win = 1
                    node.children.append(child)
                    return 1


def global_greedy(game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    frontier_nodes = [tree]
    global expanded
    return global_greedy_rec(game, frontier_nodes, color_number, heuristic, total_moves, matrix_size,
                             copy.deepcopy(visited))


def global_greedy_rec(game: MyGame, frontier: [], color_number: int, heuristic, total_moves: int, matrix_size: int,
                      visited: set):
    if len(frontier) == 0:
        return None
    global expanded
    global cant_frontier
    node = frontier.pop(0)
    visited.add(node.state.__hash__())
    if len(node.state.border_cells) + len(node.state.colored_cells) == matrix_size * matrix_size:
        cant_frontier += len(frontier) + 1
        return get_solution(node)
    if total_moves == -1 or node.state.moves < total_moves:
        for color in range(color_number):
            if color != node.state.color:
                child = get_child(node, color, game, total_moves, matrix_size, heuristic)
                child.parent = node
                if not visited.__contains__(child.state.__hash__()):
                    bisect.insort(frontier, child)
        expanded += 1
    if node.state.moves >= total_moves != -1:
        cant_frontier += 1
    return global_greedy_rec(game, frontier, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))


def a(game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    frontier_nodes = [tree]
    return a_rec(game, frontier_nodes, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited), True)


def a_rec(game: MyGame, frontier: [], color_number: int, heuristic, total_moves: int, matrix_size: int, visited: set,
          internal_count: bool):
    if len(frontier) == 0:
        return None
    global expanded
    global cant_frontier
    node = frontier.pop(0)
    visited.add(node.state.__hash__())
    if len(node.state.border_cells) + len(node.state.colored_cells) == matrix_size * matrix_size:
        if internal_count:
            cant_frontier += len(frontier) + 1
        return get_solution(node)
    if total_moves == -1 or node.state.moves < total_moves:
        for color in range(color_number):
            if color != node.state.color:
                child = get_child(node, color, game, total_moves, matrix_size, heuristic + "_cost")
                child.parent = node
                if not visited.__contains__(child.state.__hash__()):
                    bisect.insort(frontier, child)
        expanded += 1 if internal_count else 0
    if node.state.moves >= total_moves != -1:
        cant_frontier += 1 if internal_count else 0
    return a_rec(game, frontier, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited),
                 internal_count)


def get_solution(node: Node):
    node.state.win = 1
    aux_node = node
    while aux_node.parent is not None:
        aux_node_child = copy.deepcopy(aux_node)
        aux_node = copy.deepcopy(aux_node.parent)
        aux_node.state.win = 1
        aux_node.children = [aux_node_child]

    return aux_node


def get_color(number):
    colors = [
        "PINK",
        "WHITE",
        "RED",
        "GREEN",
        "BLUE",
        "YELLOW",
        "AERO_BLUE",
        "AFRICAN_VIOLET",
        "AIR_FORCE_BLUE",
        "ALLOY_ORANGE",
        "AMARANTH",
        "AMAZON",
        "AMBER",
        "ANDROID_GREEN",
        "ANTIQUE_BRASS",
        "ANTIQUE_BRONZE",
        "TAUPE",
        "AQUA",
        "LIGHT_SALMON",
        "ARSENIC",
        "ARTICHOKE",
        "ARYLIDE_YELLOW",
        "BABY_PINK",
        "BARBIE_PINK",
        "DARK_BROWN",
        "GRAY"
    ]
    return colors[number]


def get_total_cost(tree: Node):
    total_cost = 0
    aux = copy.deepcopy(tree)
    while len(aux.children) > 0:
        total_cost += 1
        aux = copy.deepcopy(aux.children[0])
    return total_cost


def shortest_path(game: MyGame, state: State, matrix_size, cant_colors):
    tree_state = copy.deepcopy(state)
    visited = set()
    tree = Node(tree_state)
    frontier_nodes = [tree]
    return get_total_cost(
        a_rec(game, frontier_nodes, cant_colors, "colors_left", -1, matrix_size, copy.deepcopy(visited), False))


def print_solution_and_get_cost(tree: Node):
    aux = copy.deepcopy(tree)
    s = get_color(aux.state.color)
    total_cost = 0
    while len(aux.children) > 0:
        total_cost += 1
        s = s.__add__(" --> ")
        aux = copy.deepcopy(aux.children[0])
        s = s.__add__(get_color(aux.state.color))
    print(s)
    return total_cost


def main():
    sys.setrecursionlimit(1000000)
    config_file = open("config.json")
    config_data = json.load(config_file)
    print(config_data)

    matrix_size = config_data["matrix_size"]
    total_moves = config_data["moves"]

    heuristic = config_data["heuristic"]
    color_number = config_data["color_number"]
    algorythm = config_data["algorythm"]

    if color_number > 26 or color_number <= 0:
        print("Se aceptan hasta 26 colores")
        exit()
    if heuristic != "cells_left" and heuristic != "colors_left" and heuristic != "shortest_path":
        print("No se reconoce la heuristica elegida")
        exit()
    screen_width = (WIDTH + MARGIN) * matrix_size + MARGIN
    screen_height = (HEIGHT + MARGIN) * (matrix_size + HEADER_COUNT) + MARGIN

    if matrix_size < color_number:
        screen_width = (WIDTH + MARGIN) * color_number + MARGIN
        screen_height = (HEIGHT + MARGIN) * (color_number + HEADER_COUNT) + MARGIN

    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, None, matrix_size, HEADER_COUNT, color_number)
    g2 = copy.deepcopy(game.grid)
    start_time = time.time()
    tree = {}
    if algorythm == "DFS":
        tree = DFS(game, color_number, total_moves, matrix_size)
    elif algorythm == "BFS":
        tree = BFS(game, color_number, total_moves, matrix_size)
    elif algorythm == "local_greedy":
        tree = local_greedy(game, color_number, heuristic, total_moves, matrix_size)
    elif algorythm == "global_greedy":
        tree = global_greedy(game, color_number, heuristic, total_moves, matrix_size)
    elif algorythm == "a*":
        tree = a(game, color_number, heuristic, total_moves, matrix_size)
    else:
        print("No se reconocio el algoritmo elegido")
        exit()
    global expanded
    global cant_frontier
    print("--- %s seconds ---" % (time.time() - start_time))
    if tree is not None:
        print("Solucion: el primero color es el color con el que se arranca")
        cost = print_solution_and_get_cost(copy.deepcopy(tree))
        print("Costo de la solucion: %d" % cost)
    else:
        print('No se ah encontrado una solucion')
    print("Se expandieron %d nodos" % expanded)
    print("cantidad de nodos fronters %d" % cant_frontier)
    print(" ")

    arcade.close_window()
    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, g2, matrix_size, HEADER_COUNT, color_number)
    arcade.run()


if __name__ == "__main__":
    main()
