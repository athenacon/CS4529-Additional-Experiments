from PIL import ImageFilter
import numpy as np
import torchvision.transforms as transforms


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 0.8 
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class TransformsSimCLR:
    def __init__(self):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.transform = transforms.Compose(
            [   
                transforms.Pad(6),
                transforms.RandomResizedCrop(size=40, scale=(0.4, 1.0)), 
                transforms.RandomApply([color_jitter], p=0.8),
                GaussianBlur(p=0.5), 
                transforms.ToTensor(), 
                # reference of normalization values:https://github.com/kuangliu/pytorch-cifar/issues/16
                transforms.Normalize(
                    mean=[0.13066047,], std = [0.30810780,]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [   
                transforms.Pad(6),
                transforms.RandomResizedCrop(size=40, scale=(0.4, 1.0)), 
                transforms.RandomApply([color_jitter], p=0.8),
                GaussianBlur(p=0.5), 
                transforms.ToTensor(), 
                # reference of normalization values:https://github.com/kuangliu/pytorch-cifar/issues/16
                transforms.Normalize(
                    mean=[0.13066047,], std = [0.30810780,]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2
