import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import matplotlib.style as mplstyle

mplstyle.use('fast')


class Regression(ABC):

    def __repr__(self):
        return f'Regression([weights={self.weights}, current_epoch={self.current_epoch}, epochs={self.epochs}])'

    def __init__(self, parameters: int, *, step: float = 0.01, epochs: int = 10000, converge_at: float = 10**-15):
        self.weights = np.random.random(parameters)
        self.step = step
        self.epochs = epochs
        self.current_epoch = 0
        self.converge_at = converge_at

        self.history: Dict[str, List] = {
            'loss': [],
            'weights': [],
        }

    @abstractmethod
    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def prep_input(self, X: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, graph: bool = True) -> Dict[str, List]:
        plotter = Plotter(self, interval=.005) if graph else None
        print_inteval = self.epochs // 10
        converged = False

        for e in range(1, self.epochs + 1):
            # Training weights
            X_pro = self.prep_input(X)
            y_hat = X_pro.dot(self.weights)
            loss, grad = self.loss(y_hat, y), self.gradient(X_pro, y_hat, y)
            self.weights -= self.step * grad
            self.history['loss'].append(loss)
            self.history['weights'].append(self.weights)
            # Display, print, converge
            if graph and not plotter.closed:
                plotter.draw_plot(X, y, epoch=e, final=False)
            if e % print_inteval == 0:
                print(f'Epoch:{e}   loss: {loss}')
            if loss <= self.converge_at:
                print(f'Epoch:{e}   loss: {loss}')
                converged = True
                break
        if plotter.closed:
            plt.close()
            plotter = Plotter(self)
        # Graph final line
        plotter.draw_plot(X, y, epoch=self.epochs, final=True)
        return self.history, converged


class Plotter:

    def __init__(self, regressor: Regression, *, interval: float = .01):
        self.regressor = regressor
        self.interval = interval
        self.plot = plt.figure()
        self.closed = False

    def draw_plot(self, X: np.ndarray, y: np.ndarray, epoch: int, final: bool = False):
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        self.plot.canvas.mpl_connect('close_event', self.on_close)

        ax1 = self.plot.add_subplot(211)
        ax1.set_title(self.regressor.weights)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2 = self.plot.add_subplot(212)
        ax2.set_title(
            f'Epochs: {epoch}/{self.regressor.epochs} Loss: {self.regressor.history["loss"][-1]}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')

        ax1.plot(X, self.regressor.predict(X), 'b-')  # plot predicted
        ax1.plot(X, y, 'ro', markersize=3)  # plot actual

        ax2.plot(list(range(
            len(self.regressor.history['loss']))), self.regressor.history['loss'], 'b--')

        if not final:
            plt.draw()
            plt.pause(self.interval)
            plt.clf()
        else:
            plt.show()

    def on_close(self, event):
        print('Figure closed!')
        self.closed = True
