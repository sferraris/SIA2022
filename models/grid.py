import arcade
import random

# Set how many rows and columns we will have
ROW_COUNT = 2 + 4
COLUMN_COUNT = 4

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 32

# Set how many moves does the user have
MOVES = 1000

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 1

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN
SCREEN_TITLE = "Array Backed Grid Example"


def is_valid(row, column):
    if row < 0 or row > ROW_COUNT - 3 or column < 0 or column >= COLUMN_COUNT:
        return False
    return True


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, moves, grid):
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

        self.grid = []
        self.colors = [arcade.color.PINK, arcade.color.WHITE, arcade.color.RED, arcade.color.GREEN, arcade.color.BLUE,
                       arcade.color.YELLOW, arcade.color.GRAY]
        self.steps = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        if grid is None:
            self.set_grid()
        else:
            self.grid = grid
        self.current_color = self.grid[ROW_COUNT - 3][0]
        self.get_initial_info()


    def set_grid(self):
        for row in range(ROW_COUNT):
            # Add an empty array that will hold each cell
            # in this row
            self.grid.append([])
            for column in range(COLUMN_COUNT):
                if row == ROW_COUNT - 1:
                    if column < 6:
                        self.grid[row].append(column)
                    else:
                        self.grid[row].append(7)
                elif row == ROW_COUNT - 2:
                    self.grid[row].append(7)
                else:
                    self.grid[row].append(random.randint(0, 5))  # Append a cell

        return self.grid

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Draw the grid
        for row in range(ROW_COUNT):
            for column in range(COLUMN_COUNT):
                # Figure out what color to draw the box [arcade.color.PINK, arcade.color.WHITE, arcade.color.RED,
                # arcade.color.GREEN, arcade.color.BLUE, arcade.color.BLACK]
                if self.grid[row][column] == 0:
                    color = arcade.color.PINK
                elif self.grid[row][column] == 1:
                    color = arcade.color.WHITE
                elif self.grid[row][column] == 2:
                    color = arcade.color.RED
                elif self.grid[row][column] == 3:
                    color = arcade.color.GREEN
                elif self.grid[row][column] == 4:
                    color = arcade.color.BLUE
                elif self.grid[row][column] == 5:
                    color = arcade.color.YELLOW
                else:
                    color = arcade.color.GRAY

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
        if row == ROW_COUNT - 1 and column < 6 and color != self.current_color and self.moves > 0:
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
            if is_valid(row + step[0], column + step[1]):
                color = grid[row][column]
                if color != grid[row + step[0]][column + step[1]]:
                    return True
        return False

    def get_initial_info(self):
        row = ROW_COUNT - 3
        column = 0
        initial_colored_cells = self.get_related_cells([[row, column]], self.current_color, self.grid, self.border_cells)

        for cell in initial_colored_cells:
            if self.is_border(cell[0], cell[1], self.grid.copy()):
                self.border_cells.append(cell)
            else:
                self.colored_cells.append(cell)

    def get_related_cells(self, cells, color, grid, border_cells):
        initial_colored_cells = []
        colored_cells_aux = cells.copy()

        while len(colored_cells_aux) > 0:
            cell = colored_cells_aux.pop()
            row = cell[0]
            column = cell[1]
            for step in self.steps:
                if is_valid(row + step[0], column + step[1]) \
                        and color == grid[row + step[0]][column + step[1]] \
                        and [row + step[0], column + step[1]] not in initial_colored_cells \
                        and [row + step[0], column + step[1]] not in cells:
                    colored_cells_aux.append([row + step[0], column + step[1]])
            if cell not in border_cells:
                initial_colored_cells.append(cell)

        return initial_colored_cells
