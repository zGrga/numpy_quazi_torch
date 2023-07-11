from numpy_quazi_torch.models.Layer import Layer
import uuid
import numpy as np

class ConvolutionLayer(Layer):
    def __init__(self,
                 kernel_size: int,
                 input_channel: int,
                 output_channel: int) -> None:
        """2D convolutional layer

        Args:
            kernel_size (int): kernel size
            input_channel (int): the number of input channels
            output_channel (int): the number of output channels
        """
        
        # define kernel size and important parameters
        self.kernel_size = (kernel_size, kernel_size, input_channel)
        self.input_channel = input_channel
        self.output_channel = output_channel
        
        # create ID of layer
        self.uuid = str(uuid.uuid1())

        # initialize weights
        self.weights = [np.random.rand(*self.kernel_size) for index in range(self.output_channel)]
        self.b = [np.random.rand() for index in range(self.output_channel)]

        # weights gradient
        self.d_weights = dict([(x, [[None]*self.input_channel, 0]) for x in range(output_channel)])

        # input gradient
        self.d_inputs = None
        self.d_inputs_counter = 0

        # bias gradient
        self.d_b = None

    def backpropagation(self, prev_delta: np.ndarray, lr: float) -> np.array:
        """Backpropagate layer

        Args:
            prev_delta (np.array): gradient from previous laayer
            lr (float): learning rate

        Returns:
            np.array: gradient on input
        """

        # backpropagation on weights
        for x_i, d_i in zip(self.last_input, prev_delta):
            # take one element from last input (H x W x C)
            # and one element from gradient (OH x OW x OC)

            d_i = np.transpose(d_i, (2, 0, 1))[:, :, :, None]
            # add dimension on gradient element and make sure that channel dimension is first (OC x OH x OW x 1)

            for index, w_i in enumerate(d_i):
                # take one channel (OH x OW x 1)

                tmp_x_i = x_i[:, :, :, None]
                # add dimension on last input (H x W x C x 1)
                
                tmp_x_i = np.transpose(tmp_x_i, (2, 0, 1, 3))
                # reorder (C x H x W x 1)

                self.d_weights[index][1] += 1

                for index_2, x_i_j in enumerate(tmp_x_i):
                    # take one channel from input (H x W x 1)
                    
                    # 2D convolution on one channel
                    if self.d_weights[index][0][index_2] is None:
                        self.d_weights[index][0][index_2] = ConvolutionLayer.__convolution__(x_i_j, w_i)
                        continue
                    
                    self.d_weights[index][0][index_2] = np.add(self.d_weights[index][0][index_2], ConvolutionLayer.__convolution__(x_i_j, w_i))

        # backpropagation on input
        storage = []
        # get one example from previous gradient
        for index_example, example in enumerate(prev_delta):
            # reorder example to be C x H x W and add dummy dimension
            # C x H x W x 1
            example = np.transpose(example, (2, 0, 1))[:, :, :, None]
            
            # create storage for output gradient for this specific example
            tmp_storage = [np.zeros((self.last_input.shape[1], self.last_input.shape[2]))]*self.last_input.shape[3]

            # iterate over each output channel
            for index, w_i in enumerate(self.weights):
                # get one channel K x K x IC
                # rotate channel to 180
                tmp_wi = np.rot90(w_i, k=2, axes=(0, 1))

                # reorder and add dummy dimension
                # IC x K x K x 1
                tmp_wi = np.transpose(tmp_wi, (2, 0, 1))[:, :, :, None]
                
                # iterate over weights channels for one specific weight
                for input_channel, tmp_w_i_j in enumerate(tmp_wi):
                    # pass through convolution and accumulate
                    conv = ConvolutionLayer.__convolution__(tmp_w_i_j, example[index], padded=True)
                    tmp_storage[input_channel] = np.add(tmp_storage[input_channel], conv)
            
            # add gradient for each input channel into storage
            storage.append(tmp_storage)

        # reorder accumulated weights to N x H x W x IC
        self.d_inputs = np.array(storage).transpose(0, 2, 3, 1)

        # backpropagation on bias
        self.d_b = [np.mean(x) for x in np.transpose(prev_delta, (3, 0, 1, 2))]

        return self.update(lr)
    
    def get_weights(self) -> list[np.ndarray]:
        to_ret = []
        for w in self.weights:
            to_ret.append(w)
        
        to_ret.append(self.b)

        return to_ret
    
    def load_weights(self, weights: list[np.ndarray]):
        for index, w in enumerate(self.weights):
            self.weights[index] = weights[index]
        
        counter = 0
        for i in range(index, len(weights)):
            self.b[counter] = weights[i]
            counter += 1
        
            
    def update(self, lr: float) -> np.ndarray:
        """Gradient on input

        Args:
            lr (float): learning rate

        Returns:
            np.ndarray: gradient on input
        """
        # update weights
        for index in range(len(self.weights)):
            # self.weights[index] = self.weights[index] - lr * (np.array(self.d_weights[index][0]).transpose(1, 2, 0) / self.d_weights[index][1]) - lr * 0.001 * self.weights[index]
            self.weights[index] = self.weights[index] - lr * (np.array(self.d_weights[index][0]).transpose(1, 2, 0) / self.last_input.shape[0]) - lr * 0.001 * self.weights[index]

        # update inputs
        input_grad = self.d_inputs / self.output_channel

        # update bias
        self.b = np.array(self.b) - lr * np.array(self.d_b)
        self.b = self.b.tolist()

        # clear weights gradients
        self.d_weights = dict([(x, [[None]*self.input_channel, 0]) for x in range(self.output_channel)])

        # clear input gradients
        self.d_inputs = None
        self.d_inputs_counter = 0

        # remove last input
        self.last_input = None

        # remove bias
        self.d_b = None

        return input_grad
    
    @staticmethod
    def __get_output_dim(x: np.ndarray, k: np.ndarray) -> tuple[int, int]:
        """Get output dimension. Stride is considered to be 1.

        Args:
            x (np.ndarray): input array
            k (np.ndarray): kernel size

        Returns:
            tuple[int, int]: height and width of output
        """
        h = int(np.floor((x.shape[0] - k) + 1))
        w = int(np.floor((x.shape[1] - k) + 1))
        return h, w
    
    @staticmethod
    def __convolution__(x: np.ndarray, k: np.ndarray, padded: bool = False, stride: int = 1) -> np.array:
        """2D convolution of x and k

        Args:
            x (np.ndarray): input
            k (np.ndarray): kernel
            padded (bool, optional): to use padded convolution. Defaults to False.
            stride (int, optional): stride. Defaults to 1.

        Returns:
            np.array: convolution results
        """
        # padding is used on backpropagation on inputs
        if padded:
            x = np.pad(x,
                       pad_width=((k.shape[0] - 1, k.shape[0] - 1), (k.shape[1] - 1, k.shape[1] - 1), (0, 0)),
                       mode='constant',
                       constant_values=0)
        
        # calculate output dimensions
        h, w = ConvolutionLayer.__get_output_dim(x, k.shape[0])

        # create output tensor
        new_size = np.zeros((h, w))
        counter_x, counter_y  = 0, 0

        for i in range(k.shape[0], x.shape[0] + 1, stride):
            for j in range(k.shape[1], x.shape[1] + 1, stride):
                window = x[(i - k.shape[0]):i, (j - k.shape[1]):j, :]
                new_size[counter_x, counter_y] = np.sum(window * k)
                counter_y += 1
            
            counter_x += 1
            counter_y = 0

        return new_size
            

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward input through layer

        Args:
            x (np.ndarray): input

        Returns:
            np.ndarray: output
        """
        self.last_input = x
        output = []

        # pass through each weight channel
        for x_i in x:
            tmp = np.transpose(np.squeeze(np.array([ConvolutionLayer.__convolution__(x_i, k) + b for k, b in zip(self.weights, self.b)])), (1, 2, 0))
            output.append(tmp)
        
        output = np.array(output)
        return output

    def __repr__(self) -> str:
        """String reprezentation of layer

        Returns:
            str: textual representation
        """
        to_ret = f'Convolutional layer (ID: {self.uuid}): \n\tkernel: {self.output_channel}x{self.kernel_size}\n\tinput_dim: {self.input_channel}\n\toutput_dim: {self.output_channel}'
        return to_ret
    
    def __str__(self) -> str:
        """String reprezentation of layer

        Returns:
            str: textual representation
        """
        return self.__repr__()