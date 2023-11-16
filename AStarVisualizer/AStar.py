
import time
import math
from itertools import product
from typing import *


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
    LIMIT = 10000

    def __str__(self):
        return f"\n****************************\n" \
               f"start={self.start}, end={self.end}, obstacles={self.obstacles}, only_cardinal={self.onlyCardinal}, box_bounds={self.box_bounds}" \
               f"\n****************************\n"

    def __init__(self, start: Tuple, end: Tuple, obstacles: List[Tuple[int, int]] = None, only_cardinal: bool = False, box_bounds: Tuple = None, debug: bool = False):
        """_summary_

        Args:
            start (Tuple): starting position
            end (Tuple): ending position
            obstacles (List[Tuple[int, int]], optional): A list of coordinates representing obstacles. Defaults to None.
            only_cardinal (bool, optional): Whether or not algorithm can only travel in four directions. Defaults to False.
            box_bounds (Tuple, optional): A tuple representing the border, x_min, x_max, y_min, y_max. Defaults to None.
            debug (bool, optional): _description_. Defaults to False.
        """
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

    def run(self) -> List:
        finished = False
        final_path = None
        while not finished:
            print(self.attempts)
            finished, final_path = self.step()
        return final_path

    def step(self) -> Union[bool, List]:
        self.attempts += 1
        if self.attempts > self.LIMIT:
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

    def get_adj_nodes(self, node: tuple) -> List:
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
    def reconstruct_path(came_from: dict, current: tuple) -> List:
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
