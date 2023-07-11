import skimage.measure
import numpy as np
from numpy_quazi_torch.models.Layer import Layer
import uuid

class MaxPooling(Layer):
    def __init__(self,
                 kernel: int) -> None:
        """Max pooling with a stride lenght the same as kernel size.

        Args:
            kernel (int): size of kernel
        """
        super().__init__()
        self.uuid = str(uuid.uuid1())
        self.kernel = kernel

        self.d_last = None
        self.delta = None

    def __calc_size(self) -> tuple[int, int]:
        """Calculate output size

        Returns:
            tuple[int, int]: height and width
        """
        h = int(np.floor((self.last.shape[1] - self.kernel) / self.kernel + 1))
        w = int(np.floor((self.last.shape[2] - self.kernel) / self.kernel + 1))
        return h, w

    def __max_pooling(self, x: np.ndarray, axis: tuple) -> int:
        """Apply max pooling over given input of dimension (kernel_size x kernel_size)

        Args:
            x (np.ndarray): input window
            axis (tuple): axis

        Returns:
            int: maximal value in a window
        """
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
    
    def get_weights(self) -> list[np.ndarray]:
        """Return list of layer weights

        Returns:
            list[np.ndarray]: empty list (no parameters)
        """
        return []


    def __call__(self, x: np.ndarray) -> np.ndarray:
        """One formward pass of the input

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: output
        """
        self.last = x

        to_return = []
        self.delta = []
        
        # select one example from the input
        for x_i in range(x.shape[0]):
            x_i = x[x_i, :, :, :]
            tmp = []
            d = []

            # pass over every channel in the input
            for c_i in range(x.shape[3]):
                c_i = x_i[:, :, c_i]
                self.shape = c_i.shape

                # apply maximization window over every channel
                max_ = skimage.measure.block_reduce(c_i, (self.kernel, self.kernel), self.__max_pooling)

                # append results into output list
                tmp.append(max_)
                d.append(self.d_last)
            
            to_return.append(np.array(tmp))
            self.delta.append(np.array(d))
        
        # pass channels to the last dimension
        self.delta = np.array(self.delta).transpose(0, 2, 3, 1)
        to_return = np.array(to_return).transpose(0, 2, 3, 1)

        return to_return
    
    def backpropagation(self, grad_last: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass (update weights and return gradient over input)

        Args:
            grad_last (np.ndarray): gradient over output
            lr (float): learning rate

        Returns:
            np.ndarray: gradient over input
        """
        # prepare output
        to_ret = np.zeros(self.delta.shape)

        to_ret = []

        # select one example gradient from the input of backpropagation (grad_last)
        # select one example from the stored location of the maximum (self.delta)
        for index, delta_prev, delta in zip(range(len(self.delta)), grad_last, self.delta):
            # flatten backpropagation input for one example
            delta_prev = np.ndarray.flatten(delta_prev)
            
            # iterate over windows
            counter_x = 0
            counter_y = 0
            value_counter = 0
            for counter_x in range(self.kernel, self.delta.shape[1] + 1, self.kernel):
                for counter_y in range(self.kernel, self.delta.shape[2] + 1, self.kernel):
                    # store input values from backpropagation on every channel in locations of the maxima
                    for c in range(delta.shape[2]):
                        delta[counter_x - self.kernel:counter_x, counter_y - self.kernel:counter_y, c] *= delta_prev[value_counter]
                        value_counter += 1

            to_ret.append(delta)

        # return results
        return np.array(to_ret)