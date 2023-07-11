import os
import cv2
import numpy as np

class DatasetFast:
    """
    Class that loades image dataset into RAM and prepare it for indexing.
    It's called "fast" beacuse the retrieving image is done from RAM.
    """
    def __init__(self,
                 path: str) -> None:
        """Initialize dataset object

        Args:
            path (str): path to the directory that consists of child directories (with images) whose names are numbered classes
        """
        self.path = path
        
        # load all possible classes in dataset
        self.classes = dict([(cls, [os.path.join(self.path, cls, x) for x in os.listdir(os.path.join(self.path, cls))]) for cls in os.listdir(path)])

        # store images into array
        self.images = []
        for dir_ in os.listdir(path):
            cls = int(dir_)

            for img_path in os.listdir(os.path.join(self.path, str(cls))):
                self.images.append((cls, cv2.imread(os.path.join(self.path, str(cls), img_path), cv2.IMREAD_GRAYSCALE)[:, :, None]))
    
    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: length
        """
        return len(self.images)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Get one data point from dataset based on index 

        Args:
            index (int): index

        Returns:
            tuple[np.ndarray, int]: tuple with image on the first index and class label on the second
        """
        cls, img = self.images[index]
        return img, cls