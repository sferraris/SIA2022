from grid import MyGame
import arcade
import anytree
from anytree import AnyNode, RenderTree

# Set how many rows and columns we will have
ROW_COUNT = 2 + 3
COLUMN_COUNT = 3

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


def get_decision_tree(game: MyGame):
    tree = Node(game.current_color, game.colored_cells.copy(), game.border_cells.copy(), game.grid.copy(), 3, 0)
    get_decision_tree_rec(tree, game)
    return tree


def get_decision_tree_rec(node: Node, game: MyGame):
    if len(node.border_cells) + len(node.colored_cells) == 9 or node.moves == 0:
        return
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
                if not game.is_border(cell[0], cell[1]):
                    border_cells.remove(cell)
                    colored_cells.append(cell)

            node.children.append(child)
            get_decision_tree_rec(child, game)


def print_tree(node: Node):
    print("holis")
    string = '{ Node: {'
    string += 'color: '
    string += str(node.color)
    string += ', cells_to_paint: '
    string += str(node.cells_to_paint)
    string +=', moves: '
    string += str(node.moves)
    string += ', total_cells_painted: '
    string += str(len(node.border_cells) + len(node.colored_cells))
    print_tree_rec(node, string)
    string += '} }'
    return string


def print_tree_rec(node: Node, string: str):
    string += ', children: ['
    if len(node.children) > 0:
        for child in node.children:
            string = '{ Node: {'
            string += 'color: '
            string += str(child.color)
            string += ', cells_to_paint: '
            string += str(child.cells_to_paint)
            string += ', moves: '
            string += str(child.moves)
            string += ', total_cells_painted: '
            string += str(len(child.border_cells) + len(child.colored_cells))
            print_tree_rec(child, string)
            string += '} },'
    string += ']'


def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, MOVES)
    tree = get_decision_tree(game)
    print(print_tree(tree))
    arcade.run()


if __name__ == "__main__":
    main()
