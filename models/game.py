import operator
import time
import bisect

from grid import MyGame
import arcade
import copy
import json

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


# def get_decision_tree(game: MyGame):
#    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves,
#                0)
#    get_decision_tree_rec(tree, game)
#    return tree
#
#
# def get_decision_tree_rec(node: Node, game: MyGame):
#    if len(node.state.border_cells) + len(node.state.colored_cells) == matrix_size * matrix_size:
#        node.state.win = 1
#        return 1
#    for color in range(6):
#        if color != node.state.color:
#            child = get_child(node, color, game)
#            node.state.children.append(child)
#
#            if child.state.moves == 0 and len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * (
#                    matrix_size - 2):
#                node.state.win = 1
#                child.state.win = 1
#            elif child.state.moves > 0:
#                return_value = get_decision_tree_rec(child, game)
#                if return_value == 1:
#                    node.state.win = 1
#    return node.state.win


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

    child_state = State(child_color, child_colored_cells, child_border_cells, child_grid, child_moves,
                        child_cells_to_paint, child_heuristic, child_cost)
    child = Node(child_state)
    return copy.deepcopy(child)


def DFS(game: MyGame, color_number: int, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    tree = Node(tree_state)
    visited = set()
    DFS_rec(tree, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))
    return tree


def DFS_rec(node: Node, game: MyGame, color_number: int, total_moves: int, matrix_size: int, visited: set):
    visited.add(node.state.__hash__())
    for color in range(color_number):
        if color != node.state.color and node.state.win == 0:
            child = get_child(node, color, game, total_moves, matrix_size, None)
            if not visited.__contains__(child.state.__hash__()):
                if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                    child.state.win = 1
                    node.state.win = 1
                    visited.add(child.state)
                    node.children.append(child)
                    return 1
                elif total_moves == -1 or child.state.moves < total_moves:
                    return_value = DFS_rec(child, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))
                    if return_value == 1:
                        node.state.win = 1
                        node.children.append(child)
                        return 1


def BFS(game: MyGame, color_number: int, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    return BFS_alg(tree, game, color_number, total_moves, matrix_size, copy.deepcopy(visited))


def BFS_alg(tree: Node, game: MyGame, color_number: int, total_moves: int, matrix_size: int, visited: set):
    bfs_queue = [tree]
    while bfs_queue:
        node = bfs_queue.pop(0)
        visited.add(node.state.__hash__())
        node.state.win = 1
        if total_moves == -1 or node.state.moves < total_moves:
            for color in range(color_number):
                if color != node.state.color:
                    child = get_child(node, color, game, total_moves, matrix_size, None)
                    if not visited.__contains__(child.state.__hash__()):
                        child.parent = node
                        node.children.append(child)
                        bfs_queue.append(child)
                        if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                            return get_solution(child)
    return None


def local_greedy(game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    local_greedy_rec(tree, game, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))
    return tree


def local_greedy_rec(node: Node, game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int,
                     visited: set):
    visited.add(node.state.__hash__())
    children_array = []
    for color in range(color_number):
        if color != node.state.color:
            child = get_child(node, color, game, total_moves, matrix_size, heuristic)
            if not visited.__contains__(child.state.__hash__()):
                bisect.insort(children_array, child)

    for child in children_array:
        if node.state.win == 0:
            if len(child.state.border_cells) + len(child.state.colored_cells) == matrix_size * matrix_size:
                child.state.win = 1
                node.state.win = 1
                node.children.append(child)
                return 1
            elif total_moves == -1 or child.state.moves < total_moves:
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
    return global_greedy_rec(game, frontier_nodes, color_number, heuristic, total_moves, matrix_size,
                             copy.deepcopy(visited))


def global_greedy_rec(game: MyGame, frontier: [], color_number: int, heuristic, total_moves: int, matrix_size: int, visited: set):
    if len(frontier) == 0:
        return None

    node = frontier.pop(0)
    visited.add(node.state.__hash__())

    if len(node.state.border_cells) + len(node.state.colored_cells) == matrix_size * matrix_size:
        return get_solution(node)
    if total_moves == -1 or node.state.moves < total_moves:
        for color in range(color_number):
            if color != node.state.color:
                child = get_child(node, color, game, total_moves, matrix_size, heuristic)
                child.parent = node
                if not visited.__contains__(child.state.__hash__()):
                    bisect.insort(frontier, child)
    return global_greedy_rec(game, frontier, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))


def a(game: MyGame, color_number: int, heuristic, total_moves: int, matrix_size: int):
    tree_state = State(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(),
                       0, 0, 0, 0)
    visited = set()
    tree = Node(tree_state)
    frontier_nodes = [tree]
    return a_rec(game, frontier_nodes, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))


def a_rec(game: MyGame, frontier: [], color_number: int, heuristic, total_moves: int, matrix_size: int, visited: set):
    if len(frontier) == 0:
        return None

    # frontier.sort(key=operator.itemgetter('heuristic_and_cost'))
    # print(f"Start Frontier: {frontier}")
    node = frontier.pop(0)
    visited.add(node.state.__hash__())
    # print(f"Lower node: {node}")

    if len(node.state.border_cells) + len(node.state.colored_cells) == matrix_size * matrix_size:
        return get_solution(node)
    if total_moves == -1 or node.state.moves < total_moves:
        for color in range(color_number):
            if color != node.state.color:
                child = get_child(node, color, game, total_moves, matrix_size, heuristic + "_cost")
                child.parent = node
                if not visited.__contains__(child.state.__hash__()):
                    bisect.insort(frontier, child)
    return a_rec(game, frontier, color_number, heuristic, total_moves, matrix_size, copy.deepcopy(visited))


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


def main():
    config_file = open("config.json")
    config_data = json.load(config_file)
    print(config_data)

    matrix_size = config_data["matrix_size"]
    total_moves = config_data["moves"]

    heuristic = config_data["heuristic"]
    color_number = config_data["color_number"]
    algorythm = config_data["algorythm"]

    screen_width = (WIDTH + MARGIN) * matrix_size + MARGIN
    screen_height = (HEIGHT + MARGIN) * (matrix_size + HEADER_COUNT) + MARGIN

    if matrix_size < color_number:
        screen_width = (WIDTH + MARGIN) * color_number + MARGIN
        screen_height = (HEIGHT + MARGIN) * (color_number + HEADER_COUNT) + MARGIN

    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, None, matrix_size, HEADER_COUNT, color_number)
    g = copy.deepcopy(game.grid)
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
    print(tree)
    print("--- %s seconds ---" % (time.time() - start_time))
    arcade.close_window()
    game = MyGame(screen_width, screen_height, SCREEN_TITLE, -1, g, matrix_size, HEADER_COUNT, color_number)

    start_time2 = time.time()
    tree2 = {}
    if algorythm == "DFS":
        tree2 = DFS(game, color_number, -1, matrix_size)
    elif algorythm == "BFS":
        tree2 = BFS(game, color_number, -1, matrix_size)
    elif algorythm == "local_greedy":
        tree2 = local_greedy(game, color_number, heuristic, -1, matrix_size)
    elif algorythm == "global_greedy":
        tree2 = global_greedy(game, color_number, heuristic, -1, matrix_size)
    elif algorythm == "a*":
        tree2 = a(game, color_number, heuristic, -1, matrix_size)

    print(tree2)
    print("--- %s seconds ---" % (time.time() - start_time2))
    # open text file
    # text_file = open("./data.json", "w")

    # write string to file
    # text_file.write(tree.__str__())

    # close file
    # text_file.close()

    arcade.close_window()
    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, g2, matrix_size, HEADER_COUNT, color_number)
    arcade.run()


if __name__ == "__main__":
    main()
