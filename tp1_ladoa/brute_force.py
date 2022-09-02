from grid import MyGame
import arcade

# Set how many rows and columns we will have
ROW_COUNT = 8
COLUMN_COUNT = 6

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
    def __init__(self, color, cells_to_paint):
        self.color = color
        self.cells_to_paint = cells_to_paint
        self.children = []


def brute_force(game: MyGame):

    parent_color = game.current_color
    options_tree = Node(parent_color)

    node = options_tree
    for move in range(MOVES):
        for color in range(COLORS - 1):
            if color != parent_color:
                options_tree.children.append(Node(color))



def main():
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, MOVES)
    #grid = game.grid.copy()
    #brute_force(grid)

    arcade.run()


if __name__ == "__main__":
    main()
