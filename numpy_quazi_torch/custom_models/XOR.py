from numpy_quazi_torch.models.Model import Model
from numpy_quazi_torch.models.LinearLayer import LinearLayer
from numpy_quazi_torch.models.RELU import RELU

class XOR(Model):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = LinearLayer(2, 8)
        self.r1 = RELU()
        self.l2 = LinearLayer(8, 8)
        self.r2 = RELU()
        self.l3 = LinearLayer(8, 4)
        self.r3 = RELU()
        self.l4 = LinearLayer(4, 2)