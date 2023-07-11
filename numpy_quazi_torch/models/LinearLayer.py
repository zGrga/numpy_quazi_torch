import numpy as np
from  numpy_quazi_torch.models.Layer import Layer
import uuid

class LinearLayer(Layer):
    def __init__(self,
                 input_size: int,
                 output_size: int) -> None:
        """Fully connected linear layer.

        Args:
            input_size (int): input dimension
            output_size (int): output dimension
        """
        
        self.input_size = input_size
        self.output_size = output_size
        # generate random weights
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5

        # generate random bias
        self.b = np.random.rand(1, self.output_size) - 0.5
    
        self.uuid = str(uuid.uuid1())

        self.d_weights = None
        self.d_inputs  = None
        self.d_b = None

    def get_weights(self) -> list:
        return [self.weights, self.b]
    
    def load_weights(self, weights: list[np.ndarray]):
        self.weights = weights[0]
        self.b = weights[1]

    def backpropagation(self, prev_delta: np.ndarray, lr: float):
        # gradient on inputs
        self.d_inputs = prev_delta@self.weights.T
        self.d_weights = (1/self.last_input.shape[0]) * (self.last_input.T@prev_delta)
        self.d_b = np.mean(prev_delta, keepdims=True)

        return self.update(lr)
    
    def update(self, lr: float):
        self.weights = self.weights - lr * self.d_weights - lr * 0.001 * self.weights
        self.b = self.b - lr*self.d_b

        d_input = self.d_inputs

        self.d_inputs = None
        self.d_weights = None

        return d_input

    def __call__(self, x: np.ndarray):
        self.last_input = x
        return x@self.weights + self.b

    def __repr__(self) -> str:
        to_ret = f'Linear layer (ID: {self.uuid}): \n\tinput_dim: {self.input_size}\n\toutput_dim: {self.output_size}'
        return to_ret
    
    def __str__(self) -> str:
        return self.__repr__()