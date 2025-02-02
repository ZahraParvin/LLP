import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import cv2

from domain_randomization_methods import domain_randomizations
import random

from os import listdir
from os.path import isfile, join


path = "./data/background_images" 

def get_background_fns(path):
    fns = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return fns


class custom_MNIST_dataset(torchvision.datasets.MNIST): 
    def __init__(
            self,
            root: str,
            train: bool = True,
            domain_randomziation_type: int = 0,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ): 
        self.domain_randomziation_type = domain_randomziation_type
        bg_fns = get_background_fns(path)
        self.bg_imgs = [[] for i in range(1000)]
        total_number_imgs = len(bg_fns)
        for i in range(1000):
            ind = random.randint(0, total_number_imgs - 1)
            self.bg_imgs[i] = cv2.imread(bg_fns[ind])
        
        super(custom_MNIST_dataset, self).__init__(root, train, transform, target_transform, download)


    def custom_transforms(self, img):
        img = cv2.resize(img, (32, 32))

        img = Image.fromarray(np.uint8(img))
        composed_transforms_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        return composed_transforms_img(img)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy().astype('uint8')

        img_3_channel = np.zeros((img.shape[0], img.shape[1], 3)).astype('uint8')
        img_3_channel[:,:,0] = img
        img_3_channel[:,:,1] = img
        img_3_channel[:,:,2] = img

        img_3_channel = domain_randomizations(img_3_channel, self.bg_imgs, self.domain_randomziation_type)
        cv2.imshow("augmented img", img_3_channel)
        cv2.waitKey(0)
        img_3_channel = self.custom_transforms(img_3_channel)

        return img_3_channel, target

