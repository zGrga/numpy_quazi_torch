import numpy as np

class Layer:
    def __init__(self) -> None:
        pass

    def backpropragation(self, lr: float):
        pass

    def update(self):
        pass

    def __call__(self, x: np.ndarray):
        pass

    def get_weights(self) -> list[np.ndarray]:
        pass

    def load_weights(self, weights: list[np.ndarray]):
        pass