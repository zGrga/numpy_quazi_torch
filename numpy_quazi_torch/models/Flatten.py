import uuid
from numpy_quazi_torch.models.Layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.uuid = str(uuid.uuid1())

    def __call__(self, x: np.ndarray):
        self.original_size = x.shape
        return x.reshape(self.original_size[0], self.original_size[1] * self.original_size[2] * self.original_size[3])

    def backpropagation(self, prev_dev: np.ndarray, **kwargs):
        return prev_dev.reshape(self.original_size)
    
    def get_weights(self) -> list[np.ndarray]:
        return []