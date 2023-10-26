from typing import Tuple
from tensorflow import keras
import keras.models
from matplotlib import pyplot as plt
import pygame
import numpy as np
from abc import ABC

WHITE = (255, 255, 255)
BLACK = (50, 50, 50)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
AQUA = (0, 100, 100)
BLUE = (0, 0, 255)

BLANK_COLOR = WHITE
HIGHLIGHT_COLOR = BLACK
BORDER_COLORS = BLACK


class TextColors(ABC):
    _codes = {
        'HEADER': '\033[95m',
        'OKBLUE': '\033[94m',
        'OKCYAN': '\033[96m',
        'OKGREEN': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
    }

    @staticmethod
    def get_codes() -> list:
        return list(TextColors._codes.keys())

    @staticmethod
    def colorize(text: str, code: str, bold: bool = False, underline: bool = False) -> str:
        ENDC = '\033[0m'
        if TextColors._codes.get(code) is None:
            raise ValueError('Color code does not exist')
        return f'{TextColors._codes.get(code)}{TextColors._codes.get("BOLD") if bold else ""}{TextColors._codes.get("UNDERLINE") if underline else ""}{text}{ENDC}'


class Tile:

    def __repr__(self) -> str:
        return f"Block([{self.coord}])"

    def __init__(self, coord, state: int = 0):
        self.coord: Tuple[int, int] = coord
        self.color: Tuple[int, int, int] = (255, 255, 255)  # default white
        self.state = state  # default state is 0

    def get_color(self):
        if self.state == 0:
            return BLANK_COLOR
        elif self.state == 1:
            return HIGHLIGHT_COLOR

    def flip_state(self) -> int:
        self.state = abs(self.state - 1)
        return self.state


class Button(Tile):
    def __repr__(self) -> str:
        return f"Button([{self.coord}])"

    def __init__(self, id, coord, state: int = 0):
        super().__init__(coord, state)
        self.id = id

    def get_color(self):
        if self.state == 0:  # calculate button
            return GREEN
        elif self.state == 1:  # reset button
            return YELLOW
        elif self.state == 2:
            return RED


class DrawingPad:

    def __init__(self, width: int = 20, tile_size: int = 30):
        # UI variables
        self.width = width
        self.tile_size = tile_size
        self.side_space = width // 5

        # Display
        self.buttons = []
        self.screen = None
        self._init_board()
        self._init_display()

        # GUI variables
        self.already_flipped = []
        self.match_state = None  # if tile state matches flip state, flip it

        # NN
        self.prediction = -1

    def _init_board(self) -> None:
        points = [[(col, row) for row in range(self.width)]
                  for col in range(self.width)]
        self.board = [list(map(lambda pos: Tile(pos), row)) for row in points]
        self.buttons = [Button('Guess', (0, self.width + self.side_space - 1), state=0),
                        Button('Show', (1, self.width +
                               self.side_space - 1), state=1),
                        Button('Reset', (2, self.width + self.side_space - 1), state=2), ]

    def _init_display(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(size=(
            (self.width + self.side_space) * self.tile_size, self.width * self.tile_size))
        pygame.display.set_caption(
            f'Drawing Pad {pygame.display.get_window_size()}, {self.width}x{self.width}')

    def reset_board(self) -> None:
        for row in self.board:
            for tile in row:
                tile.state = 0

    def get_rel_coord(self, point: tuple[int, int]) -> tuple[int, int]:
        return (point[0] // self.tile_size, point[1] // self.tile_size)[::-1]

    def in_game_board(self, point):
        return 0 <= point[0] < self.width and 0 <= point[1] < self.width

    def in_side_space(self, point):
        return 0 <= point[0] < self.width and self.width <= point[1] < self.width + self.side_space

    def get_rel_tile(self, point: tuple[int, int]) -> Tile:
        return next((tile for row in self.board for tile in row if tile.coord == point), None)

    def get_rel_button(self, point: tuple[int, int]):
        return next((button for button in self.buttons if button.coord == point), None)

    def draw_board(self) -> None:
        self.screen.fill(BLANK_COLOR)

        for button in self.buttons:  # draw buttons
            self.draw_tile(button)
        for row in self.board:  # draw tiles in board
            for tile in row:
                self.draw_tile(tile)

        # draw a border around drawing field
        rect = pygame.Rect(0, 0, self.width * self.tile_size,
                           self.width * self.tile_size)
        pygame.draw.rect(self.screen, BORDER_COLORS, rect, 3)

        # draw prediction box and text
        rect = pygame.Rect((self.width - 1 + self.side_space // 2) * self.tile_size,
                           (self.width - 1) // 2 * self.tile_size, 4 * self.tile_size, 3 * self.tile_size)
        pygame.draw.rect(self.screen, BORDER_COLORS, rect, 3)
        text_font = pygame.font.SysFont('times new roman', 30)
        text_surface = text_font.render(str(self.prediction), True, BLUE)
        text_rect = text_surface.get_rect()
        text_rect.topleft = ((self.width + self.side_space // 2)
                             * self.tile_size, self.width // 2 * self.tile_size)
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def draw_tile(self, tile: Tile, border=False) -> None:
        rect = pygame.Rect(tile.coord[::-1][0] * self.tile_size, tile.coord[::-1]
                           [1] * self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, tile.get_color(), rect, 0)  # fill
        if border:
            pygame.draw.rect(self.screen, BORDER_COLORS, rect, 1)  # border

    def get_NN_input(self) -> np.array:  # output shape of (28, 28)
        return np.array([[255 - sum(tile.get_color()) // 3 for tile in row] for row in self.board], dtype='float').reshape((1, 28, 28))


if __name__ == '__main__':
    width = 28  # must be 28 for NN
    tile_size = 10

    model = keras.models.load_model('DigitModel')
    # model = DigitClassifier()
    # model = None

    pad = DrawingPad(width, tile_size)

    while True:
        pad.draw_board()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # exit button
                pygame.quit()
                break

            # handles the mouse events
            elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                # ignore the mouse just moving around
                if event.type == pygame.MOUSEMOTION and event.buttons == (0, 0, 0):
                    continue
                coord = pad.get_rel_coord(event.pos)

                if pad.in_game_board(coord):  # events occurring within the game board
                    rel_tile = pad.get_rel_tile(coord)
                    if rel_tile is None:
                        continue

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # left click down
                            print(event.pos, coord)
                            print(rel_tile)

                        elif event.button == 3:  # right click down
                            pad.match_state = rel_tile.state
                            rel_tile.flip_state()

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:  # left click released
                            pass
                        elif event.button == 3:  # right click released
                            pad.already_flipped.clear()
                            pad.match_state = -1
                            if model is not None:
                                pred = model.predict(pad.get_NN_input())
                                pad.prediction = np.argmax(pred)
                            else:
                                print(TextColors.colorize(
                                    'No model found!', code='WARNING'))

                    elif event.type == pygame.MOUSEMOTION:  # moving mouse
                        if event.buttons == (0, 0, 1):  # right click held down
                            if rel_tile not in pad.already_flipped and rel_tile.state == pad.match_state:  # prevent double flip
                                rel_tile.flip_state()
                                pad.already_flipped.append(tile_size)

                # events occurring within the side space
                elif pad.in_side_space(coord):
                    rel_button = pad.get_rel_button(coord)
                    if rel_button is None:
                        continue

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            if rel_button.id == 'Guess':
                                if model is not None:
                                    pred = model.predict(pad.get_NN_input())
                                    pad.prediction = np.argmax(pred)
                                else:
                                    print(TextColors.colorize(
                                        'No model found!', code='WARNING'))
                            elif rel_button.id == 'Show':
                                my_img = pad.get_NN_input().reshape((28, 28, 1))
                                plt.imshow(my_img)
                                plt.show()
                            elif rel_button.id == 'Reset':
                                pad.reset_board()
                                pad.prediction = -1

        else:
            continue
        break  # will not trigger unless inner if is triggered

    print("program finished")
