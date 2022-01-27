import torch
import numpy as np 
class BaseTTA: # taken and adapted  from https://github.com/dungnb1333/global-wheat-dection-2020/blob/f4f1a9614d897d76cc2b28bf7a601004a74df15e/dataset.py#L546
    def augment(self, images):
        raise NotImplementedError

    def prepare_boxes(self, boxes):
        result_boxes = boxes.clone()
        result_boxes[:,0] = torch.min(boxes[:, [0,2]], axis=1).values
        result_boxes[:,2] = torch.max(boxes[:, [0,2]], axis=1).values
        result_boxes[:,1] = torch.min(boxes[:, [1,3]], axis=1).values
        result_boxes[:,3] = torch.max(boxes[:, [1,3]], axis=1).values
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, image):
        return image.flip(1) 

    def effdet_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return self.prepare_boxes(boxes)

class TTAVerticalFlip(BaseTTA):
    def __init__(self, image_size):
        self.image_size = image_size

    def fasterrcnn_augment(self, image):
        return image.flip(2) 

    def effdet_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes

class TTARotate90(BaseTTA):
    def __init__(self, image_size):
        self.image_size = image_size
    
    def fasterrcnn_augment(self, image):
        return torch.rot90(image, 1, (2, 3))
    
    def effdet_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.clone()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return self.prepare_boxes(res_boxes)

class TTACompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def fasterrcnn_augment(self, images):
        for transform in self.transforms:
            images = transform.fasterrcnn_augment(images)
        return images

    def effdet_augment(self, images):
        for transform in self.transforms:
            images = transform.effdet_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.clone()
        result_boxes[:,0] = torch.min(boxes[:, [0,2]], axis=1).values
        result_boxes[:,2] = torch.max(boxes[:, [0,2]], axis=1).values
        result_boxes[:,1] = torch.min(boxes[:, [1,3]], axis=1).values
        result_boxes[:,3] = torch.max(boxes[:, [1,3]], axis=1).values
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)