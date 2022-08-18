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
MOVES = 15

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
        return "{" + f" color: {get_color(self.color)}, moves: {self.moves},  win: {self.win}, children: {self.children} " + "}"

    def __repr__(self) -> str:
        return "{" + f" color: {get_color(self.color)}, moves: {self.moves},  win: {self.win}, children: {self.children} " + "}"


def get_decision_tree(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), 8, 0)
    get_decision_tree_rec(tree, game)
    return tree


def get_decision_tree_rec(node: Node, game: MyGame):
    if len(node.border_cells) + len(node.colored_cells) == COLUMN_COUNT * (ROW_COUNT - 2):
        node.win = 1
        return 1
    if node.moves == 0:
        return 0
    for color in range(6):
        if color != node.color:
            colored_cells = node.colored_cells.copy()
            border_cells = node.border_cells.copy()
            grid = node.grid.copy()
            cells_to_paint = game.get_related_cells(border_cells.copy(), color, grid, border_cells)

            child = Node(color, colored_cells, border_cells, grid, node.moves - 1, len(cells_to_paint))

            painted = colored_cells + border_cells

            for cell in painted:
                r = cell[0]
                c = cell[1]
                grid[r][c] = color

            border_cells += cells_to_paint

            for cell in border_cells:
                if not game.is_border(cell[0], cell[1], grid.copy()):
                    border_cells.remove(cell)
                    colored_cells.append(cell)

            node.children.append(child)
            return_value = get_decision_tree_rec(child, game)
            if return_value == 1:
                node.win = 1



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
