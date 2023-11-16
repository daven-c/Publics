
from typing import *
import pygame as pyg
from AStar import A_Star


class Block:

    def __str__(self):
        return "coord: " + str((self.x, self.y)) + " color: " + str(self.color) + " state: " + str(self.state)

    def __repr__(self):
        # return str(self.pos())
        return str(self.state)

    def __init__(self, coord: Tuple[int, int], color: Tuple[int, int, int], state: int):
        self.x = coord[0]
        self.y = coord[1]
        self.color = color
        self.state = state  # 0 for Colors/Whitespace 1 for obstacle, -1 for special

    def pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def flip_state(self):
        if self.state != -1:  # -1 exclude from flipping
            self.state = abs(self.state - 1)
            self.color = (0, 0, 0) if self.state == 1 else (255, 255, 255)


class Visualizer:

    def __init__(self, *, width: int = 800, block_width: int = 40, fps: int = 300):
        """_summary_

        Args:
            width (int, optional): width of display in pixels. Defaults to 800.
            block_width (int, optional): width of each tile in pixels. Defaults to 40.
            fps (int, optional): speed of game. Defaults to 300.
        """
        # Pygame display variables
        self.width = width
        self.block_width = block_width
        self.n_tiles = width // block_width
        self.game_window = None
        self.speed = fps  # defined by run()
        self._init_display()

        # Game Variables
        self.starting_tile = (0, 0)
        self.ending_tile = (self.n_tiles - 1, self.n_tiles - 1)
        self.grid = self._blank_grid()

    def run(self):
        # Get the coordinates of each obstacles and initialize finder
        obst = [b.pos() for row in self.grid for b in row if b.state == 1]
        finder = A_Star(start=self.starting_tile, end=self.ending_tile, obstacles=obst,
                        only_cardinal=True, box_bounds=(0, self.n_tiles, 0, self.n_tiles))

        # Run the AStar and render each step
        finished = False
        final_path = None
        while not finished:
            finished, final_path = finder.step()
            for r, c in finder.allNodes:
                if (r, c) not in (self.starting_tile, self.ending_tile, *obst):
                    self.grid[r][c].color = (200, 200, 100)
            for r, c in finder.openSet:
                if (r, c) not in (self.starting_tile, self.ending_tile):
                    self.grid[r][c].color = (0, 0, 100)
            for r, c in final_path:
                if (r, c) not in (self.starting_tile, self.ending_tile):
                    self.grid[r][c].color = (0, 0, 100)
            self.draw_frame()
            self.clear_path()

        # Final state
        for r, c in final_path:
            if (r, c) not in (self.starting_tile, self.ending_tile):
                self.grid[r][c].color = (0, 150, 0)
            self.draw_frame()
        print('Complete')

    def _init_display(self):
        if pyg.display.get_init() is False:
            pyg.init()
            self.game_window = pyg.display.set_mode(
                (self.width, self.width))
            self.fps = pyg.time.Clock()
            pyg.display.set_caption(f"{(self.width, self.width)}")

    # Set all blocks to state 0 and white
    def _blank_grid(self) -> List[List[Block]]:
        tiles = [[Block((r, c), (255, 255, 255), 0)
                  for c in range(self.n_tiles)] for r in range(self.n_tiles)]
        # Change starting block
        tiles[self.starting_tile[0]][self.starting_tile[1]] = Block(
            (self.starting_tile[0], self.starting_tile[1]), (0, 255, 0), -1)
        # Change end block
        tiles[self.ending_tile[0]][self.ending_tile[1]] = Block(
            (self.ending_tile[0], self.ending_tile[1]), (255, 0, 0), -1)
        return tiles

    def clear_path(self):
        for row in self.grid:
            for block in row:
                if block.state == 0:
                    block.color = (255, 255, 255)

    def draw_block(self, block: Block):
        rect = pyg.Rect(block.y * self.block_width, block.x *
                        self.block_width, self.block_width, self.block_width)
        pyg.draw.rect(self.game_window, block.color, rect, 0)
        pyg.draw.rect(self.game_window, (0, 0, 0), rect, 1)

    def draw_frame(self):
        self.game_window.fill((255, 255, 255))
        for row in self.grid:
            for block in row:
                self.draw_block(block)
        pyg.display.update()
        self.fps.tick(self.speed)

    def in_grid(self, coord: Tuple[int, int]) -> bool:
        return all(map(lambda x: 0 <= x < self.n_tiles, coord))

    def get_rel_coord(self, point: Tuple[int, int]) -> Tuple[int, int]:
        return point[1] // self.block_width, point[0] // self.block_width

    # Returns the associated block based on coordinate position
    def get_rel_block(self, point: Tuple[int, int]) -> Block:
        if point[0] < self.n_tiles and point[1] < self.n_tiles:
            return [block for row in self.grid for block in row if block.pos() == point][0]
        return None


if __name__ == "__main__":

    """
    Space to run,
    Backspace to restart,
    Left click to print coord,
    Right click to place or erase obstacle
    """

    width: int = 600
    block_width: int = 20
    fps: int = 300

    visualizer = Visualizer(width=width, block_width=block_width, fps=fps)

    flipped = []
    match_state = 0  # makes sure that if you begin right click on
    while True:
        for event in pyg.event.get():
            if event.type == pyg.QUIT:  # exit button
                pyg.quit()
                quit()

            elif event.type == pyg.KEYUP:
                if event.key == pyg.K_SPACE:  # Space to start
                    print('start')
                    visualizer.clear_path()
                    visualizer.run()
                elif event.key == pyg.K_BACKSPACE:  # Backspace to reset
                    print('reset')
                    visualizer.grid = visualizer._blank_grid()

            elif hasattr(event, 'pos'):
                pos = visualizer.get_rel_coord(event.pos)
                block = visualizer.get_rel_block(pos)

                if event.type == pyg.MOUSEBUTTONDOWN:  # left click to print pos right to flip blocks
                    if event.button == 1:  # left click to interact
                        print(block)

                    elif event.button == 3:  # right click to flip blocks
                        if visualizer.in_grid(pos):
                            block.flip_state()
                            match_state = block.state
                            flipped.append(block)

                if event.type == pyg.MOUSEMOTION:  # moving mouse
                    if visualizer.in_grid(pos):
                        if event.buttons == (0, 0, 1):  # right click down
                            # prevent double flip and flipping same state blocks
                            if block not in flipped and block.state != match_state:
                                block.flip_state()
                                flipped.append(block)
                        # mouse release clears flipped
                        if event.buttons == (0, 0, 0):
                            flipped.clear()

        visualizer.draw_frame()
