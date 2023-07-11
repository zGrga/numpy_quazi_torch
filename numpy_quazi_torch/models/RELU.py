from numpy_quazi_torch.models.Layer import Layer
import numpy as np
import uuid

class RELU(Layer):
    def __init__(self) -> None:
        """RELU activation function
        """
        self.uuid = str(uuid.uuid1())
        self.positions = None

    def __call__(self, x: np.ndarray):
        self.positions = np.array(x)
        self.positions[self.positions<=0] = 0
        self.positions[self.positions>0] = 1
        return self.positions * x
    
    def backpropagation(self, prev_delta: np.array, **kwargs):
        to_ret = self.positions
        self.positions = None
        return to_ret * prev_delta
    
    def __repr__(self) -> str:
        return f'RELU (ID: {self.uuid})'
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def get_weights(self) -> list[np.ndarray]:
        return []