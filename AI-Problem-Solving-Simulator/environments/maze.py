
class Maze:

    def __init__(self, grid):

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.start = None
        self.goal = None

        self.find_start_goal()

    def find_start_goal(self):

        for row in range(self.rows):

            for col in range(self.cols):

                if self.grid[row][col] == "S":
                    self.start = (row, col)

                elif self.grid[row][col] == "G":
                    self.goal = (row, col)

    def is_valid(self, row, col):

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and self.grid[row][col] != "#"
        )

    def get_neighbors(self, position):

        row, col = position

        directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]

        neighbors = []

        for dr, dc in directions:

            new_row = row + dr
            new_col = col + dc

            if self.is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))

        return neighbors

    def display(self):

        for row in self.grid:
            print(" ".join(row))


if __name__ == "__main__":

    grid = [
        ["S", ".", ".", "#", "."],
        ["#", "#", ".", "#", "."],
        [".", ".", ".", ".", "."],
        [".", "#", "#", "#", "."],
        [".", ".", ".", "G", "."]
    ]

    maze = Maze(grid)

    print("Maze:")
    maze.display()

    print("\nStart Position:", maze.start)
    print("Goal Position:", maze.goal)

    print("\nNeighbours of Start:")
    print(maze.get_neighbors(maze.start))
