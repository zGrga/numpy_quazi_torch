import numpy as np
from numpy_quazi_torch.models.Layer import Layer
import os

class Model:
    """Represents one model
    """
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Pass the input through the models layers

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: output
        """
        for key in self.__dict__:
            if isinstance(self.__dict__[key], Layer):
                x = self.__dict__[key](x)
        return x

    def backpropagation(self, delta: np.ndarray, lr: float) -> None:
        """Backpropagate over all layers

        Args:
            delta (np.ndarray): gradient over output
            lr (float): learning rate
        """
        for key in reversed(self.__dict__.keys()):
            if isinstance(self.__dict__[key], Layer):
                delta = self.__dict__[key].backpropagation(delta, lr=lr)

    def store_parameters(self, path: str) -> None:
        """Store all layers weights

        Args:
            path (str): output path
        """
        for index, key in enumerate(self.__dict__):
            if isinstance(self.__dict__[key], Layer):
                pth = os.path.join(path, f'{index}')
                os.makedirs(pth)

                for counter, w in enumerate(self.__dict__[key].get_weights()):
                    np.save(os.path.join(pth, f'{counter}.npy'), w)

    def load_parameters(self, path: str) -> None:
        """Load all layers weights

        Args:
            path (str): path to the weights
        """
        for index, key in enumerate(self.__dict__):
            if isinstance(self.__dict__[key], Layer):
                pth = os.path.join(path, f'{index}')

                to_load_list = []
                for to_load in [os.path.join(pth, x) for x in os.listdir(pth)]:
                    to_load_list.append(np.load(to_load))

                self.__dict__[key].load_weights(to_load_list)