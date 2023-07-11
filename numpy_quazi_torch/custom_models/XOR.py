from numpy_quazi_torch.models.Model import Model
from numpy_quazi_torch.models.LinearLayer import LinearLayer
from numpy_quazi_torch.models.RELU import RELU

class XOR(Model):
    def __init__(self) -> None:
        """
        Simple XOR on 2 dimensional inputs
        """
        super().__init__()

        # first LINEAR layer
        self.l1 = LinearLayer(2, 8)

        # RELU activation function
        self.r1 = RELU()

        # second LINEAR layer
        self.l2 = LinearLayer(8, 8)

        # RELU activation function
        self.r2 = RELU()

        # third LINEAR layer
        self.l3 = LinearLayer(8, 4)

        # RELU activation function
        self.r3 = RELU()

        # pre output LINEAR layer
        self.l4 = LinearLayer(4, 2)