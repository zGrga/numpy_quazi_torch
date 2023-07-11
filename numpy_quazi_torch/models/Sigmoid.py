import numpy as np
from  numpy_quazi_torch.models.Layer import Layer
import uuid

class Sigmoid(Layer):
    def __init__(self) -> None:
        """Sigmoid activation function
        """
        self.uuid = str(uuid.uuid1())

    def __sigmoid(self, x: np.ndarray) -> np.array:
        """Sigmoid function

        Args:
            x (np.ndarray): input

        Returns:
            np.array: sigmoid output
        """
        return 1/(1 + np.exp(-x))
    
    def __call__(self, x: np.ndarray):
        self.last = x
        return self.__sigmoid(x)

    def backpropagation(self, prev_delta: np.array, **kwargs):
        return prev_delta * self.__sigmoid(self.last)*(1 - self.__sigmoid(self.last))
    
    def get_weights(self) -> list[np.ndarray]:
        return []