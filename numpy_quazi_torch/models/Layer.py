import numpy as np

class Layer:
    def __init__(self) -> None:
        """Layer initialization.
        """
        pass

    def backpropagation(self, input_gradient: np.ndarray, lr: float):
        """One backpropagation step.

        Args:
            input_gradient (np.ndarray): gradient over output
            lr (float): learning rate
        """
        pass

    def update(self):
        """Update weights
        """
        pass

    def __call__(self, x: np.ndarray):
        """Pass input through layer.

        Args:
            x (np.ndarray): input
        """
        pass

    def get_weights(self) -> list[np.ndarray]:
        """Generate list of the layer weights

        Returns:
            list[np.ndarray]: list of weights
        """
        pass

    def load_weights(self, weights: list[np.ndarray]):
        """Load layer weights from list

        Args:
            weights (list[np.ndarray]): list of loaded weights
        """
        pass