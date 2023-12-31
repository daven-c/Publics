import numpy as np
from ParentClass import Regression
from typing import Union, List


class LinearReg(Regression):

    def __init__(self, *, step: float = 0.01, epochs: int = 10000, converge_at: float = 10**-5):
        super().__init__(parameters=2, step=step, epochs=epochs, converge_at=converge_at)

    # MSE
    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        loss = np.mean(np.square((y - y_hat)))
        return loss

    def gradient(self, X: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        gradient = -2 * \
            np.mean((y - y_hat).reshape((X.shape[0], 1)) * X, axis=0)
        return gradient

    @staticmethod
    def prep_input(X: np.ndarray):
        return np.vstack((X, np.ones((1, len(X))))).T

    def predict(self, X: Union[List, np.ndarray, float]) -> np.ndarray:
        if type(X) is not np.ndarray:
            if type(X) is float:
                X = list(X)
            X = np.asarray(X)
        X_pro = self.prep_input(X)
        y_hat = X_pro.dot(self.weights)
        return y_hat
