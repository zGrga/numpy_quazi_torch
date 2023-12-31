from numpy_quazi_torch.models.Model import Model
from numpy_quazi_torch.models.ConvolutionLayer import ConvolutionLayer
from numpy_quazi_torch.models.RELU import RELU
from numpy_quazi_torch.models.MaxPooling import MaxPooling
from numpy_quazi_torch.models.Flatten import Flatten
from numpy_quazi_torch.models.Sigmoid import Sigmoid
from numpy_quazi_torch.models.LinearLayer import LinearLayer
import os

class MNIST(Model):
    def __init__(self) -> None:
        """
        MNIST model that recognizes handwritten digits.
        """
        super().__init__()

        # first layer that flattens 28 x 28 image into 784 x 1
        self.l_08 = Flatten()

        # first LINEAR layer trat transfers 784 x 1 vector into 10 x 1
        self.l_09 = LinearLayer(input_size=784, output_size=10)

        # RELU activation function
        self.l_10 = RELU()

        # second LINEAR layer
        self.l_11 = LinearLayer(input_size=10, output_size=10)