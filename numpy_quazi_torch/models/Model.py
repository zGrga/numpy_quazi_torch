import numpy as np
from numpy_quazi_torch.models.Layer import Layer
import os

class Model:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray):
        for key in self.__dict__:
            if isinstance(self.__dict__[key], Layer):
                x = self.__dict__[key](x)
        return x

    def backpropagation(self, delta: np.ndarray, lr: float):
        for key in reversed(self.__dict__.keys()):
            if isinstance(self.__dict__[key], Layer):
                delta = self.__dict__[key].backpropagation(delta, lr=lr)

    def store_parameters(self, path: str):
        for index, key in enumerate(self.__dict__):
            if isinstance(self.__dict__[key], Layer):
                pth = os.path.join(path, f'{index}')
                os.makedirs(pth)

                for counter, w in enumerate(self.__dict__[key].get_weights()):
                    np.save(os.path.join(pth, f'{counter}.npy'), w)

    def load_parameters(self, path: str):
         for index, key in enumerate(self.__dict__):
            if isinstance(self.__dict__[key], Layer):
                pth = os.path.join(path, f'{index}')

                to_load_list = []
                for to_load in [os.path.join(pth, x) for x in os.listdir(pth)]:
                    to_load_list.append(np.load(to_load))

                self.__dict__[key].load_weights(to_load_list)