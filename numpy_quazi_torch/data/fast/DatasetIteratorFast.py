from numpy_quazi_torch.data.fast.DatasetFast import DatasetFast
import random
import numpy as np

class DatasetIteratorFast:
    def __init__(self,
                 dataset: DatasetFast,
                 batch_size: int,
                 train_flag: bool = True,
                 train_size: float = 0.7) -> None:
        
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_flag = train_flag

        shuffle = list(range(0, len(self.dataset)))

        train_shuffle = random.sample(shuffle, k=int(train_size * len(shuffle)))
        valid_shuffle = list(set(shuffle) - set(train_shuffle))

        random.shuffle(train_shuffle)
        random.shuffle(valid_shuffle)
        
        self.train_grupations = self.get_grupation(train_shuffle, self.batch_size)
        self.valid_grupations = self.get_grupation(valid_shuffle, self.batch_size)
        

        self.train_data = []
        for group in self.train_grupations:
            x, y = [], []

            for indx in group:
                x_, y_ = self.dataset[indx]
                y_t = [0 for x in self.dataset.classes.keys()]
                y_t[y_] = 1
                y_ = y_t
                
                x.append(x_)
                y.append(y_)

            self.train_data.append((np.array(x, dtype=np.float32) / 255, np.array(y, dtype=np.float32)))


        self.valid_data = []
        for group in self.valid_grupations:
            x, y = [], []

            for indx in group:
                x_, y_ = self.dataset[indx]
                y_t = [0 for x in self.dataset.classes.keys()]
                y_t[y_] = 1
                y_ = y_t
                
                x.append(x_)
                y.append(y_)

            self.valid_data.append((np.array(x, dtype=np.float32) / 255, np.array(y, dtype=np.float32)))


        self.train_counter = 0
        self.valid_counter = 0

    def __len__(self):
        if self.train_flag:
            return len(self.train_data)
        
        return len(self.valid_data)
    
    def reset_counters(self):
        self.train_counter = 0
        self.valid_counter = 0
    
    def to_train(self, train: bool = False):
        self.train_flag = train
    
    def get_grupation(self, list_, n):
        for i in range(0, len(list_), n):
            yield list_[i:i + n]

    def get(self):
        if self.train_flag:
            to_send = self.train_data[self.train_counter]
            self.train_counter += 1
            return to_send
        
        to_send = self.valid_data[self.valid_counter]
        self.valid_counter += 1
        return to_send
    