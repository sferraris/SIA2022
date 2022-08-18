from grid import MyGame
import arcade
import copy
import anytree
from anytree import AnyNode, RenderTree

# Set how many rows and columns we will have
ROW_COUNT = 2 + 4
COLUMN_COUNT = 4

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 32

# Set how many moves does the user have
MOVES = 8

COLORS = 6

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "Brute Force"


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

    def __str__(self) -> str:
        return "{" + f" color: {get_color(self.color)},moves: {self.moves},  win: {self.win}, total_cells: {len(self.border_cells) + len(self.colored_cells)}, children: {self.children} " + "}"

    def __repr__(self) -> str:
        return "{" + f" color: {get_color(self.color)},moves: {self.moves},  win: {self.win}, total_cells: {len(self.border_cells) + len(self.colored_cells)}, children: {self.children} " + "}"


def get_decision_tree(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), MOVES, 0)
    get_decision_tree_rec(tree, game)
    return tree


def get_decision_tree_rec(node: Node, game: MyGame):
    if len(node.border_cells) + len(node.colored_cells) == COLUMN_COUNT * (ROW_COUNT - 2):
        node.win = 1
        return 1
    for color in range(6):
        if color != node.color:
            child_color = color
            cells_to_paint = game.get_related_cells(copy.deepcopy(node.border_cells), color, node.grid, copy.deepcopy(node.border_cells))
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
            node.children.append(child)

            if child_moves == 0 and len(child_border_cells) + len(child_colored_cells) == COLUMN_COUNT * (ROW_COUNT - 2):
                node.win = 1
                child.win = 1
            elif child_moves > 0:
                return_value = get_decision_tree_rec(child, game)
                if return_value == 1:
                    node.win = 1
    return node.win



def print_tree(node: Node):
    print("holis")
    string = '{ Node: {'
    string += 'color: '
    string += get_color(node.color)
   # string += ', cells_to_paint: '
   # string += str(node.cells_to_paint)
    string +=', moves: '
    string += str(node.moves)
    string += ', win: '
    string += str(node.win)
  #  string += ', total_cells_painted: '
  #  string += str(len(node.border_cells) + len(node.colored_cells))
    string += print_tree_rec(node)
    string += '} }'
    return string

def get_color(number):
    colors = ["PINK", "WHITE", "RED", "GREEN", "BLUE",
                   "YELLOW"]
    return colors[number]


def print_tree_rec(node: Node):
    aux = ', children: ['
    if len(node.children) > 0:
        for child in node.children:
            aux += '{ Node: {'
            aux += 'color: '
            aux += get_color(node.color)
          #  aux += ', cells_to_paint: '
          #  aux += str(child.cells_to_paint)
            aux += ', moves: '
            aux += str(child.moves)
            aux += ', win: '
            aux += str(node.win)
          #  aux += ', total_cells_painted: '
          #  aux += str(len(child.border_cells) + len(child.colored_cells))
            aux += print_tree_rec(child)
            aux += '} },'
    aux += ']'
    return aux

def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, MOVES, None)
    g = copy.deepcopy(game.grid)

    tree = get_decision_tree(game)
   # print(tree)
    # open text file
    text_file = open("./data.json", "w")

    # write string to file
    text_file.write(tree.__str__())

    # close file
    text_file.close()




    arcade.close_window()
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, MOVES, g)
    arcade.run()


if __name__ == "__main__":
    main()
