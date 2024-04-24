from PIL import ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1 
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [    
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0, saturation=0.1, hue=0
                        )
                    ],
                    p=0.8,    
                    ),  
                GaussianBlur(p=1.0),
                transforms.ToTensor(), 
                 
            ]
        )
        self.transform_prime = transforms.Compose(
            [    
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,      ),          
                GaussianBlur(p=0.1), 
                transforms.ToTensor(), 
                 
            ]
        )
 

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2