from numpy_quazi_torch.data.fast.DatasetFast import DatasetFast
import random
import numpy as np

class DatasetIteratorFast:
    def __init__(self,
                 dataset: DatasetFast,
                 batch_size: int,
                 train_flag: bool = True,
                 train_size: float = 0.7) -> None:
        """Dataset itertor that handles batching process and splitting 
        dataset into a train and valid subset

        Args:
            dataset (DatasetFast): dataset object
            batch_size (int): size of batch
            train_flag (bool, optional): training mode. Defaults to True.
            train_size (float, optional): training subset size. Defaults to 0.7.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # flag that says in which mode currently the DatasetIterrator is working
        # needed because the same iterrator is handeling both train and valid subsets
        self.train_flag = train_flag

        # create list of indexes
        shuffle = list(range(0, len(self.dataset)))

        # randomly select traning index subset
        train_shuffle = random.sample(shuffle, k=int(train_size * len(shuffle)))
        # leftovers use as a index subset used for valdiation
        valid_shuffle = list(set(shuffle) - set(train_shuffle))

        # create random orders
        random.shuffle(train_shuffle)
        random.shuffle(valid_shuffle)
        
        # create batches
        self.train_grupations = self.get_grupation(train_shuffle, self.batch_size)
        self.valid_grupations = self.get_grupation(valid_shuffle, self.batch_size)
        
        # create train batches
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

            # transform loaded images to np.float32 format and normalize values [0.0 to 1.0]
            self.train_data.append((np.array(x, dtype=np.float32) / 255, np.array(y, dtype=np.float32)))

        # create validation batches
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

            # transform loaded images to np.float32 format and normalize values [0.0 to 1.0]
            self.valid_data.append((np.array(x, dtype=np.float32) / 255, np.array(y, dtype=np.float32)))

        # counters used for iterative fetching of batches
        self.train_counter = 0
        self.valid_counter = 0

    def __len__(self) -> int:
        """The length of the dataset (dependes on the working mode, can be train or valid mode).

        Returns:
            int: length
        """
        if self.train_flag:
            return len(self.train_data)
        
        return len(self.valid_data)
    
    def reset_counters(self):
        """Reset counters, iterate batches from zero
        """
        self.train_counter = 0
        self.valid_counter = 0
    
    def to_train(self, train: bool = False):
        """Change working mode, if train=False, the output will be batches from the validation subset.
        If train=True, the output will be batches from the train subset.

        Args:
            train (bool): train mode. Defaults to False.
        """
        self.train_flag = train
    
    def get_grupation(self, list_: list, n: int) -> list:
        """Get batched grupations (of size n) based on the passedlist.

        Args:
            list_ (list): list
            n (int): size of batch

        Returns:
            list: batched list

        Yields:
            Iterator[list]: batched list
        """
        for i in range(0, len(list_), n):
            yield list_[i:i + n]

    def get(self) -> tuple[np.array, np.array]:
        """Get the next batch in a sequence.

        Returns:
            tuple[np.array, np.array]: batch
        """
        if self.train_flag:
            to_send = self.train_data[self.train_counter]
            self.train_counter += 1
            return to_send
        
        to_send = self.valid_data[self.valid_counter]
        self.valid_counter += 1
        return to_send
    