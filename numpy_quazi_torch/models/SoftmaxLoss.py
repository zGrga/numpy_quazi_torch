import numpy as np
from numpy_quazi_torch.models.Layer import Layer
import uuid
import scipy

class SoftmaxLoss(Layer):
    def __init__(self) -> None:
        self.uuid = str(uuid.uuid1())

    def __repr__(self) -> str:
        return f'SoftmaxLoss (ID: {self.uuid})'

    def __call__(self, x: np.ndarray):
        return scipy.special.softmax(x, axis=1)
    
    def loss(self, y_hat: np.ndarray, y):
        return -1 * np.mean(np.log(self(y_hat) + 1e-10)*y)
    
    def backpropagation(self, y_hat: np.ndarray, y: np.ndarray):
        to_ret = (y_hat - y)
        self.y = None
        self.last = None
        return to_ret
    
    def get_weights(self) -> list[np.ndarray]:
        return []