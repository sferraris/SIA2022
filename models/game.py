import operator
import bisect

from grid import MyGame
import arcade
import copy
import json

# Set how many rows and columns we will have
HEADER_COUNT = 2
matrix_size = 6

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 32

# Set how many moves does the user have
total_moves = 2

COLORS = 6

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

# Do the math to figure out our screen dimensions
SCREEN_TITLE = "Game"


class Node:
    def __init__(self, color, colored_cells, border_cells, grid, moves, cells_to_paint):
        self.color = color
        self.cells_to_paint = cells_to_paint
        self.colored_cells = colored_cells
        self.border_cells = border_cells
        self.grid = grid
        self.children = []
        self.moves = moves
        self.win = 0
        self.parent = None
        self.cells_left = matrix_size * matrix_size - (len(self.border_cells) + len(self.colored_cells))
        self.heuristic_and_cost = (total_moves - moves) + self.cells_left

    def __str__(self) -> str:
        # return "{" + f" color: {get_color(self.color)},moves: {self.moves},  win: {self.win}, total_cells: {len(self.border_cells) + len(self.colored_cells)}, children: {self.children} " + "}"
        return f"h(c): {self.heuristic_and_cost}, color: {get_color(self.color)} -> {self.children} "

    def __repr__(self) -> str:
        return f"h(c): {self.heuristic_and_cost}, color: {get_color(self.color)} -> {self.children} "

    def __getitem__(self, key):
        return getattr(self, key)

    def __gt__(self, other):
        if self.heuristic_and_cost == other.heuristic_and_cost:
            return self.cells_left.__gt__(other.cells_left)
        return self.heuristic_and_cost.__gt__(other.heuristic_and_cost)

    def __lt__(self, other):
        if self.heuristic_and_cost == other.heuristic_and_cost:
            return self.cells_left.__lt__(other.cells_left)
        return self.heuristic_and_cost.__lt__(other.heuristic_and_cost)


def get_decision_tree(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves, 0)
    get_decision_tree_rec(tree, game)
    return tree


def get_decision_tree_rec(node: Node, game: MyGame):
    if len(node.border_cells) + len(node.colored_cells) == matrix_size * matrix_size:
        node.win = 1
        return 1
    for color in range(6):
        if color != node.color:
            child = get_child(node, color, game)
            node.children.append(child)

            if child.moves == 0 and len(child.border_cells) + len(child.colored_cells) == matrix_size * (
                    matrix_size - 2):
                node.win = 1
                child.win = 1
            elif child.moves > 0:
                return_value = get_decision_tree_rec(child, game)
                if return_value == 1:
                    node.win = 1
    return node.win


def get_child(node: Node, color: int, game: MyGame):
    child_color = color
    cells_to_paint = game.get_related_cells(copy.deepcopy(node.border_cells), color, node.grid,
                                            copy.deepcopy(node.border_cells))
    child_cells_to_paint = len(cells_to_paint)
    child_grid = copy.deepcopy(node.grid)
    child_moves = node.moves - 1

    painted = copy.deepcopy(node.colored_cells) + copy.deepcopy(node.border_cells)
    for cell in painted:
        r = cell[0]
        c = cell[1]
        child_grid[r][c] = color

    child_colored_cells = copy.deepcopy(node.colored_cells)
    child_border_cells = copy.deepcopy(node.border_cells) + cells_to_paint

    for cell in child_border_cells:
        if not game.is_border(cell[0], cell[1], child_grid):
            child_border_cells.remove(cell)
            child_colored_cells.append(cell)

    child = Node(child_color, child_colored_cells, child_border_cells, child_grid, child_moves, child_cells_to_paint)
    return copy.deepcopy(child)


def DFS(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves, 0)
    DFS_rec(tree, game)
    return tree


def DFS_rec(node: Node, game: MyGame):
    for color in range(6):
        if color != node.color and node.win == 0:
            child = get_child(node, color, game)
            if len(child.border_cells) + len(child.colored_cells) == matrix_size * matrix_size:
                child.win = 1
                node.win = 1
                node.children.append(child)
                return 1
            elif child.moves > 0:
                return_value = DFS_rec(child, game)
                if return_value == 1:
                    node.win = 1
                    node.children.append(child)
                    return 1


def BFS(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves, 0)
    return BFS_alg(tree, game)


def BFS_alg(tree: Node, game: MyGame):
    bfs_queue = [tree]
    while bfs_queue:
        node = bfs_queue.pop(0)
        node.win = 1
        if node.moves > 0:
            for color in range(6):
                if color != node.color:
                    child = get_child(node, color, game)
                    child.parent = node
                    node.children.append(child)
                    bfs_queue.append(child)
                    if len(child.border_cells) + len(child.colored_cells) == matrix_size * matrix_size:
                        return get_solution(child)
    return None


def greedy(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves, 0)
    greedy_rec(tree, game)
    return tree


def greedy_rec(node: Node, game: MyGame):
    children_array = []
    for color in range(6):
        if color != node.color:
            child = get_child(node, color, game)
            children_array.append(child)
    children_array.sort(key=operator.itemgetter('cells_left'))

    for child in children_array:
        if node.win == 0:
            if len(child.border_cells) + len(child.colored_cells) == matrix_size * matrix_size:
                child.win = 1
                node.win = 1
                node.children.append(child)
                return 1
            elif child.moves > 0:
                return_value = greedy_rec(child, game)
                if return_value == 1:
                    node.win = 1
                    node.children.append(child)
                    return 1


def a(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), total_moves, 0)
    frontier_nodes = [tree]
    return a_rec(game, frontier_nodes)


def a_rec(game: MyGame, frontier: []):
    if len(frontier) == 0:
        return None

    #frontier.sort(key=operator.itemgetter('heuristic_and_cost'))
    print(f"Start Frontier: {frontier}")
    node = frontier.pop(0)
    print(f"Lower node: {node}")
    print(f"End Frontier: {frontier}")

    if len(node.border_cells) + len(node.colored_cells) == matrix_size * matrix_size:
        return get_solution(node)
    if node.moves > 0:
        for color in range(6):
            if color != node.color:
                child = get_child(node, color, game)
                child.parent = node
                bisect.insort(frontier, child)
    return a_rec(game, frontier)


def get_solution(node: Node):
    node.win = 1
    aux_node = node
    while aux_node.parent is not None:
        aux_node_child = copy.deepcopy(aux_node)
        aux_node = copy.deepcopy(aux_node.parent)
        aux_node.win = 1
        aux_node.children = [aux_node_child]
    return aux_node


def get_color(number):
    colors = ["PINK", "WHITE", "RED", "GREEN", "BLUE", "YELLOW"]
    return colors[number]


def main():
    config_file = open("config.json")
    config_data = json.load(config_file)
    print(config_data)
    print(config_data["matrix_size"])

    matrix_size = config_data["matrix_size"]
    total_moves = config_data["moves"]
    heuristic = config_data["heuristic"]
    color_number = config_data["color_number"]

    screen_width = (WIDTH + MARGIN) * matrix_size + MARGIN
    screen_height = (HEIGHT + MARGIN) * (matrix_size + HEADER_COUNT) + MARGIN

    if matrix_size < color_number:
        screen_width = (WIDTH + MARGIN) * color_number + MARGIN
        screen_height = (HEIGHT + MARGIN) * (color_number + HEADER_COUNT) + MARGIN

    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, None, matrix_size, HEADER_COUNT, color_number, heuristic, "asd")
    g = copy.deepcopy(game.grid)

    #tree = a(game)
    # tree = get_decision_tree(game)
    #print(tree)
    # open text file
    # text_file = open("./data.json", "w")

    # write string to file
    # text_file.write(tree.__str__())

    # close file
    # text_file.close()

    arcade.close_window()
    game = MyGame(screen_width, screen_height, SCREEN_TITLE, total_moves, g, matrix_size, HEADER_COUNT, color_number, heuristic, "asd")
    arcade.run()


if __name__ == "__main__":
    main()
