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
        super().__init__()

        # self.l_08 = Flatten()
        # self.l_09 = LinearLayer(input_size=784, output_size=512)
        # self.l_10 = RELU()
        # self.l_11 = LinearLayer(input_size=512, output_size=256)
        # self.l_12 = RELU()
        # self.l_13 = LinearLayer(input_size=256, output_size=64)
        # self.l_14 = RELU()
        # self.l_15 = LinearLayer(input_size=64, output_size=10)
        # # self.l_20 = Sigmoid()




        self.l_08 = Flatten()
        self.l_09 = LinearLayer(input_size=784, output_size=10)
        self.l_10 = RELU()
        self.l_11 = LinearLayer(input_size=10, output_size=10)
        # self.l_20 = Sigmoid()