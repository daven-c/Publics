import math
from itertools import product
import time
import pygame
from typing import *
from collections import namedtuple


def memoize(func):  # Example memoize wrapper
    cache = {}

    def call_foo(*args, **kwargs):  # Parameter must accept original functions parameter
        if args in cache:
            return cache[args]
        else:
            cache[args] = func(*args)
            return cache[args]

    return call_foo


class A_Star:
    limit = 10000

    def __str__(self):
        return f"\n****************************\n" \
               f"start={self.start}, end={self.end}, obstacles={self.obstacles}, only_cardinal={self.onlyCardinal}, box_bounds={self.box_bounds}" \
               f"\n****************************\n"

    def __init__(self, start: tuple, end: tuple, obstacles: list = None, only_cardinal: bool = False, box_bounds: tuple = None, debug: bool = False):
        self.allNodes = []
        self.start = start
        self.end = end
        self.obstacles = list(set(obstacles)) if obstacles is not None else []
        self.onlyCardinal = only_cardinal
        self.box_bounds = box_bounds if box_bounds is not None and len(
            box_bounds) != 0 else None
        self.debug = debug
        self.__confirmArgs()

        # Algo variables
        self.closedSet = self.obstacles.copy()
        self.openSet = [self.start]
        self.cameFrom = dict()
        self.gScore = {self.start: 0}
        self.fScore = {self.start: self.heuristicFunction(
            self.start, self.end)}
        self.run_time = time.time() if self.debug else 0
        self.attempts = 0

    @staticmethod
    def print_debug(event, indent: int = 1, split: bool = False, **kwargs):
        if not split:
            print('\t' * indent + f'- {event}:', kwargs)

    def __confirmArgs(self):
        assert isinstance(self.start, tuple) and len(self.start) == 2
        assert isinstance(self.end, tuple) and len(self.end) == 2
        assert isinstance(self.obstacles, list)
        assert isinstance(self.onlyCardinal, bool)
        if self.box_bounds is not None:
            assert (isinstance(self.box_bounds, tuple)) and (
                len(self.box_bounds) == 4)
            assert (self.box_bounds[0] <= self.box_bounds[1]) and (
                self.box_bounds[2] <= self.box_bounds[3])

    def getPath(self) -> list:
        finished = False
        final_path = None
        while not finished:
            print(self.attempts)
            finished, final_path = self.step()
        return final_path

    def step(self) -> Union[bool, List]:
        self.attempts += 1
        if self.attempts > self.limit:
            if self.debug:
                A_Star.print_debug('Failed', indent=0, start=self.start, end=self.end,
                                   calc_time=f"{round(time.time() - self.run_time, 2)}s")
            return True, []
        if len(list(self.fScore.values())) == 0:
            return True, []
        # get lowest in list of all fCosts
        minCost = min(list(self.fScore.values()))
        # get point with the lowest cost
        current = next(
            (x for x in self.fScore if self.fScore[x] == minCost), ())
        if current == self.end:  # END CASE
            path = self.reconstruct_path(self.cameFrom, current)
            if self.debug:
                A_Star.print_debug('Success', indent=0, start=self.start, end=self.end, path_len=len(
                    path), calc_time=f"{round(time.time() - self.run_time, 2)}s")
            return True, path

        self.closedSet.append(current)
        self.openSet.remove(current)

        neighbors = self.get_adj_nodes(current)
        # all nodes appending
        notIn = [x for x in neighbors if x not in self.allNodes]
        self.allNodes.extend(notIn)
        if current not in self.allNodes:
            self.allNodes.append(current)
        for neighbor in neighbors:
            if neighbor in self.closedSet:
                continue

            # append neighbor into openSet
            tentative_gScore = self.gScore[current] + \
                self.heuristicFunction(current, neighbor)

            if neighbor not in self.openSet:
                self.openSet.append(neighbor)
                self.gScore[neighbor] = self.heuristicFunction(
                    neighbor, self.start)
                self.fScore[neighbor] = self.heuristicFunction(neighbor, self.start) + self.heuristicFunction(neighbor,
                                                                                                              self.end)
            elif tentative_gScore >= self.gScore[neighbor]:
                continue

            self.cameFrom[neighbor] = current
            self.gScore[neighbor] = tentative_gScore
            self.fScore[neighbor] = self.gScore[neighbor] + \
                self.heuristicFunction(neighbor, self.end)

        self.gScore.pop(current)
        self.fScore.pop(current)
        return False, self.reconstruct_path(self.cameFrom, current)

    def get_adj_nodes(self, node: tuple) -> list:
        neighbors = list(product([node[0] - 1, node[0], node[0] + 1],  # generate 3x3
                                 [node[1] - 1, node[1], node[1] + 1]))
        neighbors.remove(node)  # remove center point
        if self.onlyCardinal:  # only cardinal points
            neighbors = list(
                filter(lambda n: n[0] == node[0] or n[1] == node[1], neighbors))
        if self.box_bounds is not None:  # check in bounds if there is bounds
            neighbors = [n for n in neighbors if (self.box_bounds[0] <= n[0] < self.box_bounds[1]) and (
                self.box_bounds[2] <= n[1] < self.box_bounds[3])]
        return neighbors

    @staticmethod
    def reconstruct_path(came_from: dict, current: tuple) -> list:
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    @staticmethod
    @memoize
    def heuristicFunction(point1: tuple, point2: tuple,
                          multiplier=10) -> int:  # employs pythagorean theorem to find direct distance
        return int(math.sqrt(((point2[1] - point1[1]) ** 2) + ((point2[0] - point1[0]) ** 2)) * multiplier)


class Block:

    def __str__(self):
        return "coord: " + str((self.x, self.y)) + " color: " + str(self.color) + " state: " + str(self.state)

    def __repr__(self):
        # return str(self.pos())
        return str(self.state)

    def __init__(self, coord: tuple[int, int], color: tuple[int, int, int], state: int):
        self.x = coord[0]
        self.y = coord[1]
        self.color = color
        self.state = state  # 0 for Colors/Whitespace 1 for obstacle, -1 for special

    def pos(self) -> tuple[int, int]:
        return self.x, self.y

    def flip_state(self):
        if self.state != -1:  # -1 exclude from flipping
            self.state = abs(self.state - 1)
            self.color = (0, 0, 0) if self.state == 1 else (255, 255, 255)


class Visualizer:

    def __init__(self, *, width: int = 800, block_width: int = 40, fps: int = 60):
        # Pygame display variables
        self.width = width
        self.block_width = block_width
        self.n_tiles = width // block_width
        self.game_window = None
        self.speed = fps  # defined by run()
        self._init_display()

        # Game Variables
        self.starting_tile = (self.n_tiles // 2, self.n_tiles // 2)
        self.ending_tile = (self.n_tiles - 1, self.n_tiles - 1)
        self.grid = self._init_grid()

    def _init_display(self):
        if pygame.display.get_init() is False:
            pygame.init()
            self.game_window = pygame.display.set_mode(
                (self.width, self.width))
            self.fps = pygame.time.Clock()
            pygame.display.set_caption(f"{(self.width, self.width)}")

    def _init_grid(self) -> List[List[Block]]:
        tiles = [[Block((r, c), (255, 255, 255), 0)
                  for c in range(self.n_tiles)] for r in range(self.n_tiles)]
        tiles[self.starting_tile[0]][self.starting_tile[1]] = Block(
            (self.starting_tile[0], self.starting_tile[1]), (0, 255, 0), -1)
        tiles[self.ending_tile[0]][self.ending_tile[1]] = Block(
            (self.ending_tile[0], self.ending_tile[1]), (255, 0, 0), -1)
        return tiles

    def clear_path(self):
        for row in self.grid:
            for block in row:
                if block.state == 0:
                    block.color = (255, 255, 255)

    def draw_block(self, block: Block):
        rect = pygame.Rect(block.y * self.block_width, block.x *
                           self.block_width, self.block_width, self.block_width)
        pygame.draw.rect(self.game_window, block.color, rect, 0)
        pygame.draw.rect(self.game_window, (0, 0, 0), rect, 1)

    def draw_frame(self):
        self.game_window.fill((255, 255, 255))
        for row in self.grid:
            for block in row:
                self.draw_block(block)
        pygame.display.update()
        self.fps.tick(self.speed)

    def draw_algo(self):
        obst = [b.pos() for row in self.grid for b in row if b.state == 1]
        finder = A_Star(start=self.starting_tile, end=self.ending_tile, obstacles=obst,
                        only_cardinal=True, box_bounds=(0, self.n_tiles, 0, self.n_tiles))
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
            self.draw_frame()
            self.clear_path()

        # Final state
        for r, c in final_path:
            if (r, c) not in (self.starting_tile, self.ending_tile):
                self.grid[r][c].color = (0, 150, 0)
            self.draw_frame()
        print('Complete')

    def in_grid(self, coord: tuple[int, int]) -> bool:
        return all(map(lambda x: 0 <= x < self.n_tiles, coord))

    def get_rel_coord(self, point: tuple[int, int]) -> tuple[int, int]:
        return point[1] // self.block_width, point[0] // self.block_width

    def get_rel_block(self, point: tuple[int, int]) -> Block:
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
    fps: int = 60

    visualizer = Visualizer(width=width, block_width=block_width, fps=fps)

    flipped = []
    match_state = 0  # makes sure that if you begin right click on
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # exit button
                pygame.quit()
                quit()

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:  # Space to start
                    print('start')
                    visualizer.clear_path()
                    visualizer.draw_algo()
                elif event.key == pygame.K_BACKSPACE:  # Backspace to reset
                    print('reset')
                    visualizer.grid = visualizer._init_grid()

            elif hasattr(event, 'pos'):
                pos = visualizer.get_rel_coord(event.pos)
                block = visualizer.get_rel_block(pos)

                if event.type == pygame.MOUSEBUTTONDOWN:  # left click to print pos right to flip blocks
                    if event.button == 1:  # left click to interact
                        print(block)

                    elif event.button == 3:  # right click to flip blocks
                        if visualizer.in_grid(pos):
                            block.flip_state()
                            match_state = block.state
                            flipped.append(block)

                if event.type == pygame.MOUSEMOTION:  # moving mouse
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
