import numbers
from torchvision import transforms
from torchvision.transforms import functional as F
import torch
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torchvision.transforms

# These Transformations are replaced by the Albumentations package

class NRandomCrop(object):
    def __init__(self, size, n=1):
        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        w, h = F.get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [torch.randint(0, h - th + 1, size=(1, )).item() for i in range(n)]
        j_list = [torch.randint(0, w - tw + 1, size=(1, )).item() for i in range(n)]

        return i_list, j_list, th, tw

    def __call__(self, img):
        crops = []
        i, j, h, w = self.get_params(img, self.size, self.n)
        for k in range(len(i)):
            new_crop = F.crop(img,i[k], j[k], h, w)
            crops.append(new_crop)
        return tuple(crops)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class NOrderedCrop(object): # TODO Not implemented yet
    def __init__(self, size, n=1):
        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))
        self.n = n

    @staticmethod
    def get_params(img, output_size, n, overlap=0):
        w, h = F.get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        ind = 0
        points_w = []
        points_h = []

        for i in range(np.int32(np.sqrt(n))):
            for j in range(np.int32(np.sqrt(n))):
                points_h.append(i*((h-th)/(np.int32(np.sqrt(n)-1))))
                points_w.append(j*((w-tw)/(np.int32(np.sqrt(n)-1))))
                ind += 1
        if n == 5:
            points_w.append(int((w/2)-tw/2))
            points_h.append(int((h/2)-th/2))
        return points_h, points_w, th, tw

    def __call__(self, img):
        crops = []
        i, j, h, w = self.get_params(img, self.size, self.n)
        for k in range(len(i)):
            # for l in range(len(j)):
            new_crop = F.crop(img,i[k], j[k], h, w)
            crops.append(new_crop)
        
        # Visualitation for debugging..
        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # for k in range(len(i)):
        #     rect = patches.Rectangle((i[k],j[k]),h,w,linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # ii=0
        # for crop in crops:
        #     plt.imshow(crop)
        #     plt.savefig(f'/home/Behrendt/crop{ii}.png')
        #     ii=ii+1

        return tuple(crops)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.RC = transforms.RandomCrop(size)
        self.size = size 

    def __call__(self, image, target):
        i, j, h, w  = self.RC.get_params(image,(self.size,self.size))
        image = F.crop(image, i, j, h, w)

        bbox = target["boxes"]
        if (i > bbox[:,[0]]).any().item() :
            i = int(bbox[:,[0]].min().item())
        if (j > bbox[:,[1]]).any().item() :
            j = int(bbox[:,[1]].min().item())

        bbox[:, [0, 2]] = bbox[:, [0, 2]] - i
        bbox[:, [1, 3]] = bbox[:, [1, 3]] - j

        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size 

    def __call__(self, image, target):
        i = int((1024 - self.size) / 2)
        j = int((1024 - self.size) / 2)
        image = F.center_crop(image, self.size)

        bbox = target["boxes"]

        bbox[:, [0, 2]] = bbox[:, [0, 2]] - i
        bbox[:, [1, 3]] = bbox[:, [1, 3]] - j
        for j, box in enumerate(bbox): 
            for k, coord in enumerate(box) : 
                if coord < 0:
                    bbox[j][k] = 0
                    
        return image, target

class RandomBrightness(object):
    def __init__(self, brightness= 0, contrast=0, saturation=0,hue=0  ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, image, target):

        image = transforms.ColorJitter(brightness=self.brightness,contrast=self.contrast,saturation=self.saturation,hue=self.hue)(image)
        return image, target

# class Grayscale(object):
#     def __init__(self, num_output_channels=3):
#         self.inchannels = num_output_channels
#     def __call__(self, image, target):
#         image = transforms.Grayscale(3)(self.inchannels)
#         return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target



def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size