import uuid
from numpy_quazi_torch.models.Layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self) -> None:
        """Flattten tensor (leaves the first dimension intact)
        """
        super().__init__()
        self.uuid = str(uuid.uuid1())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Flatten input

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: flattened output
        """
        self.original_size = x.shape
        return x.reshape(self.original_size[0], self.original_size[1] * self.original_size[2] * self.original_size[3])

    def backpropagation(self, prev_dev: np.ndarray, **kwargs) -> np.ndarray:
        """Backpropagation step

        Args:
            prev_dev (np.ndarray): gradient over output

        Returns:
            np.ndarray: gradient over input
        """
        return prev_dev.reshape(self.original_size)
    
    def get_weights(self) -> list[np.ndarray]:
        return []