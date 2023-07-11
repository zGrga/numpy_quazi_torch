from typing import Any
import numpy as np

class ExponentialScheduler:
    def __init__(self,
                 start_lr: float,
                 stop_learning_rate: float,
                 step: float) -> None:
        """Reduce learning rate exponentialy by multiplying learning rate with step
        Args:
            start_lr (float): starting learning rate
            stop_learning_rate (float): stopping learning rate
            step (float): step
        """
        
        self.lr = start_lr
        self.stop_learning_rate = stop_learning_rate
        self.step = step
        self.first = True

    def __call__(self) -> float:
        """Get learning rate

        Returns:
            float: learning rate
        """
        return self.lr
    
    def update(self) -> None:
        """Update learning rate
        """
        self.lr = self.step * self.lr

        