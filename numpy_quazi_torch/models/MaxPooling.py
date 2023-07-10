import skimage.measure
import numpy as np
from numpy_quazi_torch.models.Layer import Layer
import uuid

class MaxPooling(Layer):
    def __init__(self,
                 kernel: int) -> None:
        super().__init__()
        self.uuid = str(uuid.uuid1())
        self.kernel = kernel

        self.d_last = None
        self.delta = None

    def __calc_size(self):
        h = int(np.floor((self.last.shape[1] - self.kernel) / self.kernel + 1))
        w = int(np.floor((self.last.shape[2] - self.kernel) / self.kernel + 1))
        return h, w

    def __max_pooling(self, x: np.ndarray, axis: tuple):
        h, w = self.__calc_size()
        zeros = np.zeros(self.shape)
        to_ret = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                maximum = np.max(x[i][j])
                argmax = np.where(x[i][j] == maximum)
                m_x = argmax[0][0]
                m_y = argmax[1][0]

                to_ret[i][j] = maximum
                
                zeros[i*2 + m_x][j*2 + m_y] = 1

        self.d_last = zeros
        return to_ret


    def __call__(self, x: np.ndarray):
        self.last = x

        to_return = []
        self.delta = []

        for x_i in range(x.shape[0]):
            x_i = x[x_i, :, :, :]
            tmp = []
            d = []

            for c_i in range(x.shape[3]):
                c_i = x_i[:, :, c_i]
                self.shape = c_i.shape
                max_ = skimage.measure.block_reduce(c_i, (self.kernel, self.kernel), self.__max_pooling)
                tmp.append(max_)
                d.append(self.d_last)
            
            to_return.append(np.array(tmp))
            self.delta.append(np.array(d))
        
        self.delta = np.array(self.delta).transpose(0, 2, 3, 1)
        to_return = np.array(to_return).transpose(0, 2, 3, 1)

        # print(self.delta)

        return to_return
    
    def backpropagation(self, grad_last: np.ndarray, lr: float):
        to_ret = np.zeros(self.delta.shape)

        for index, delta_prev, delta in zip(range(len(self.delta)), grad_last, self.delta):
            indices = np.where(delta == 1)
            delta_prev = np.ndarray.flatten(delta_prev)

            dim_1, dim_2, dim_3 = indices

            for i, j, k, value in zip(dim_1, dim_2, dim_3, delta_prev):
                to_ret[index][i][j][k] = value
        return to_ret