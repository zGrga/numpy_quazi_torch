import os
import cv2

class DatasetFast:
    def __init__(self,
                 path: str) -> None:
        
        self.path = path
        self.classes = dict([(cls, [os.path.join(self.path, cls, x) for x in os.listdir(os.path.join(self.path, cls))]) for cls in os.listdir(path)])

        self.images = []
        for dir_ in os.listdir(path):
            cls = int(dir_)

            for img_path in os.listdir(os.path.join(self.path, str(cls))):
                self.images.append((cls, cv2.imread(os.path.join(self.path, str(cls), img_path), cv2.IMREAD_GRAYSCALE)[:, :, None]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        cls, img = self.images[index]
        return img, cls