import copy

import arcade
import random

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 32

# Set how many moves does the user have
MOVES = 1000

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

TOTAL_COLORS = 25


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, moves, grid, matrix_size, header_count, color_count):
        """
        Set up the application.
        """

        super().__init__(width, height, title)

        # Create a 2 dimensional array. A two-dimensional
        # array is simply a list of lists.
        self.colored_cells = []
        self.border_cells = []
        self.current_color = None
        self.moves = moves

        self.matrix_size = matrix_size
        self.header_count = header_count
        self.color_count = color_count

        self.real_size = self.matrix_size
        if self.matrix_size < self.color_count:
            self.real_size = self.color_count

        self.grid = []

        self.colors = [
            arcade.color.PINK,
            arcade.color.WHITE,
            arcade.color.RED,
            arcade.color.GREEN,
            arcade.color.BLUE,
            arcade.color.YELLOW,
            arcade.color.AERO_BLUE,
            arcade.color.AFRICAN_VIOLET,
            arcade.color.AIR_FORCE_BLUE,
            arcade.color.ALLOY_ORANGE,
            arcade.color.AMARANTH,
            arcade.color.AMAZON,
            arcade.color.AMBER,
            arcade.color.ANDROID_GREEN,
            arcade.color.ANTIQUE_BRASS,
            arcade.color.ANTIQUE_BRONZE,
            arcade.color.TAUPE,
            arcade.color.AQUA,
            arcade.color.LIGHT_SALMON,
            arcade.color.ARSENIC,
            arcade.color.ARTICHOKE,
            arcade.color.ARYLIDE_YELLOW,
            arcade.color.BABY_PINK,
            arcade.color.BARBIE_PINK,
            arcade.color.DARK_BROWN,
            arcade.color.GRAY
        ]
        self.steps = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        if grid is None:
            self.set_grid()
        else:
            self.grid = grid
        self.current_color = self.grid[self.real_size - 1][0]
        self.get_initial_info()

    def set_grid(self):
        for row in range(self.real_size + self.header_count):
            # Add an empty array that will hold each cell
            # in this row
            self.grid.append([])
            for column in range(self.real_size):
                if row == self.real_size + self.header_count - 1:
                    if column < self.color_count:
                        self.grid[row].append(column)
                    else:
                        self.grid[row].append(TOTAL_COLORS)
                elif row == self.real_size:
                    self.grid[row].append(TOTAL_COLORS)
                else:
                    if column < self.matrix_size and self.real_size > row > self.real_size - self.matrix_size - 1:
                        self.grid[row].append(random.randint(0, self.color_count - 1))
                    else:
                        self.grid[row].append(TOTAL_COLORS)

        return self.grid

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Draw the grid
        for row in range(self.real_size + self.header_count):
            for column in range(self.real_size):
                # Figure out what color to draw the box [arcade.color.PINK, arcade.color.WHITE, arcade.color.RED,
                # arcade.color.GREEN, arcade.color.BLUE, arcade.color.BLACK]
                color = self.colors[self.grid[row][column]]

                # Do the math to figure out where the box is
                x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
                y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2

                # Draw the box
                arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the user presses a mouse button.
        """
        # Change the x/y screen coordinates to grid coordinates
        column = int(x // (WIDTH + MARGIN))
        row = int(y // (HEIGHT + MARGIN))

        print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")
        # Make sure we are on-grid. It is possible to click in the upper right
        # corner in the margin and go to a grid location that doesn't exist
        color = column
        if row == self.real_size + self.header_count - 1 and column < self.color_count and color != self.current_color and self.moves > 0:
            self.moves -= 1
            print(f"Moves Left {self.moves}")
            related_cells = self.get_related_cells(self.border_cells.copy(), color, self.grid, self.border_cells)
            cells_to_paint = self.colored_cells + self.border_cells
            for cell in cells_to_paint:
                r = cell[0]
                c = cell[1]
                self.grid[r][c] = color

            self.border_cells += related_cells
            self.current_color = color

            for cell in self.border_cells:
                if not self.is_border(cell[0], cell[1], self.grid.copy()):
                    self.border_cells.remove(cell)
                    self.colored_cells.append(cell)

    def is_border(self, row, column, grid):
        for step in self.steps:
            if self.is_valid(row + step[0], column + step[1]):
                color = grid[row][column]
                if color != grid[row + step[0]][column + step[1]]:
                    return True
        return False

    def get_initial_info(self):
        row = self.real_size - 1
        column = 0
        initial_colored_cells = self.get_related_cells([[row, column]], self.current_color, self.grid,self.border_cells)

        for cell in initial_colored_cells:
            if self.is_border(cell[0], cell[1], self.grid.copy()):
                self.border_cells.append(cell)
            else:
                self.colored_cells.append(cell)

    def get_related_cells(self, cells, color, grid, border_cells):
        initial_colored_cells = []
        colored_cells_aux = copy.deepcopy(cells)

        while len(colored_cells_aux) > 0:
            cell = colored_cells_aux.pop()
            row = cell[0]
            column = cell[1]
            for step in self.steps:
                if self.is_valid(row + step[0], column + step[1]) \
                        and color == grid[row + step[0]][column + step[1]] \
                        and [row + step[0], column + step[1]] not in initial_colored_cells \
                        and [row + step[0], column + step[1]] not in cells:
                    colored_cells_aux.append([row + step[0], column + step[1]])
            if cell not in border_cells and cell not in initial_colored_cells:
                initial_colored_cells.append(cell)

        return initial_colored_cells

    def is_valid(self, row, column):
        if row < self.real_size - self.matrix_size or row > self.real_size - 1 or column < 0 or column >= self.matrix_size:
            return False
        return True
