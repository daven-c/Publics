import pygame
import random


class Visualizer:
    @staticmethod
    def getMethods():
        return [x[1:-4] for x in dir(Visualizer) if x.endswith("Sort")]

    def __init__(self, win_dims: tuple[int, int] = (400, 400)):
        self.n = None  # defined by run()
        self.data = None  # defined by run()
        self.method = None  # defined by run()
        self.comparisons = 0

        self.win_x, self.win_y = win_dims
        self.tile_width = None  # defined by run()
        self.game_window = None
        self.speed = None  # defined by run()
        self._init_display()

    def _init_display(self):
        if pygame.display.get_init() is False:
            pygame.init()
            self.game_window = pygame.display.set_mode(
                (self.win_x, self.win_y))
            self.fps = pygame.time.Clock()

    def draw_frame(self, target_idx: int = None, switch_idx: int = None, final: bool = False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # exit button
                pygame.quit()
                quit()
        self.game_window.fill((45, 45, 45))
        pygame.display.set_caption(
            f"{(self.win_x, self.win_y)}, type: {self.method}, comparisons: {self.comparisons}")
        for i, v in enumerate(self.data):
            pygame.draw.rect(self.game_window, (255, 255, 255), pygame.Rect(
                i * self.tile_width, self.win_y - (v / self.n) * self.win_y, self.tile_width, self.win_y))
        if target_idx is not None:
            pygame.draw.rect(self.game_window, (255, 255, 0), pygame.Rect(target_idx * self.tile_width,
                             self.win_y - (self.data[target_idx] / self.n) * self.win_y, self.tile_width, self.win_y))
        if switch_idx is not None:
            pygame.draw.rect(self.game_window, (255, 0, 0), pygame.Rect(switch_idx * self.tile_width,
                             self.win_y - (self.data[switch_idx] / self.n) * self.win_y, self.tile_width, self.win_y))
        if final:
            highlight_speed = self.n * 2
            for i, v in enumerate(self.data[::-1]):
                pygame.draw.rect(self.game_window, (255, 255, 0), pygame.Rect(
                    (self.n - i - 1) * self.tile_width, self.win_y - (v / self.n) * self.win_y, self.tile_width, self.win_y))
                pygame.display.update()
                self.fps.tick(highlight_speed)
                pygame.draw.rect(self.game_window, (255, 255, 255), pygame.Rect(
                    (self.n - i - 1) * self.tile_width, self.win_y - (v / self.n) * self.win_y, self.tile_width, self.win_y))
            for i, v in enumerate(self.data):
                pygame.draw.rect(self.game_window, (0, 0, 255), pygame.Rect(
                    i * self.tile_width, self.win_y - (v / self.n) * self.win_y, self.tile_width, self.win_y))
                pygame.display.update()
                self.fps.tick(highlight_speed)
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # exit button
                        return
        pygame.display.update()
        self.fps.tick(self.speed)

    def run(self, method: str, n: int = 50, speed: int = 60):
        assert method in Visualizer.getMethods()
        assert n <= self.win_x
        assert self.win_x % n == 0

        self.n = n
        self.data = list(range(1, self.n + 1))
        random.shuffle(self.data)
        self.method = method
        self.comparisons = 0
        self.tile_width = self.win_x / self.n
        self.speed = speed

        print(
            f'run({[f"{k}={v}" for k, v in locals().items() if type(v) in (str, int)]})'
        )
        # get sorting method reference
        sorter = getattr(self, "_" + method + "Sort")
        sorter()  # run method
        print(f"finished with {self.comparisons} comparisons")

    # ascending
    def _selectionSort(self):
        for i in range(len(self.data) - 1):
            for j in range(i + 1, len(self.data)):
                self.comparisons += 1
                self.draw_frame(target_idx=i, switch_idx=j)
                if self.data[i] > self.data[j]:
                    self.data[i], self.data[j] = self.data[j], self.data[i]
        self.draw_frame(final=True)

    # ascending
    def _insertionSort(self):
        for i in range(1, len(self.data)):
            key = self.data[i]
            j = i - 1
            while j >= 0 and key < self.data[j]:
                self.comparisons += 1
                self.draw_frame(target_idx=i, switch_idx=j)
                self.data[j + 1] = self.data[j]
                j -= 1
            self.data[j + 1] = key
        self.draw_frame(final=True)

    # ascending
    def _bubbleSort(self):
        for i in range(len(self.data) - 1, 0, -1):
            for j in range(i):
                if self.data[j] > self.data[j + 1]:
                    self.comparisons += 1
                    self.draw_frame(target_idx=j, switch_idx=j+1)
                    self.data[j], self.data[j +
                                            1] = self.data[j + 1], self.data[j]
        self.draw_frame(final=True)

    def _mergeSort(self, l=None, r=None):
        def _mergehelper(start, mid, end):
            start2 = mid + 1

            # If the direct merge is already sorted
            self.comparisons += 1
            if self.data[mid] <= self.data[start2]:
                return

            # Two pointers to maintain start
            # of both self.dataays to merge
            while start <= mid and start2 <= end:
                self.draw_frame(target_idx=end, switch_idx=start)
                # If element 1 is in right place
                self.comparisons += 1
                if self.data[start] <= self.data[start2]:
                    start += 1
                else:
                    value = self.data[start2]
                    index = start2

                    # Shift all the elements between element 1
                    # element 2, right by 1.
                    while index != start:
                        self.data[index] = self.data[index - 1]
                        index -= 1

                    self.data[start] = value

                    # Update all the pointers
                    start += 1
                    mid += 1
                    start2 += 1

        l = 0 if l is None else l
        r = self.n - 1 if r is None else r
        if l < r:
            m = l + (r - l) // 2

            # Sort first and second halves
            self._mergeSort(l, m)
            self._mergeSort(m + 1, r)
            _mergehelper(l, m, r)
        if self.data == sorted(self.data.copy()):
            self.draw_frame(final=True)
