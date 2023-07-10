from typing import Any
import numpy as np

class LinearScheduler:
    def __init__(self,
                 start_lr: float,
                 stop_learning_rate: float,
                 step: float) -> None:
        
        self.lr = start_lr
        self.stop_learning_rate = stop_learning_rate
        self.step = step
        self.first = True

    def __call__(self):
        return self.lr
    
    def update(self):
        self.lr = self.step * self.lr

        