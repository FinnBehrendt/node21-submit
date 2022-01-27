from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from skimage.transform import resize
from PIL import Image
import pandas as pd
import os
from src.utils import custom_transforms
import ctypes
import multiprocessing as mp
import SimpleITK as sitk
import src.utils.custom_transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random 
"""
Everything related to the data sets
Parts of the Dataset classes and functions are taken from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
"""


class DataSet(Dataset):
   
    def __init__(self, csv, cfg, transform=None, data_aug=None):
        super(DataSet, self).__init__()
        self.cfg = cfg
        self.csv = csv
        self.set = csv['set'].values[0]
        self.labels = csv.label
        self.imgpath = csv.Path
        self.data_aug = data_aug
        self.transform = get_transforms(cfg,self.set)
        self.preload = cfg.preload
        # Preload Data to RAM
        if cfg.preload :
            shared_array_base = mp.Array(ctypes.c_float, len(self.labels)  * 1024 * 1024)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            self.preLoadList = shared_array.reshape(len(self.labels),1024,1024)
            # self.preLoadList_lat = self.preLoadList.copy()
            self.use_cache = False
    def __len__(self):
        return len(self.labels)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def get_labels(self):
        return self.labels

    def _process(self,idx):
        imgid = self.csv['Path'].iloc[idx]
        imgid = self.cfg.path.train.images + imgid
        img = sitk.ReadImage(imgid)
        img = sitk.GetArrayFromImage(img)
        # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        # img = cv2.convertScaleAbs(img)
        img = img.astype(np.float32) / np.max(img) 
        if self.cfg.get('invertIntensity',False):
            img = img.max() - img
        # print(img.max(), img.min(), img.mean())
        # img = Image.fromarray((np.asarray(img)/np.max(img)))
        
        if self.data_aug is not None:
            img = self.data_aug(img)
        
        return img 
    
    def _process_labels(self,idx):
        num_objs = 1
        if self.csv['x2'].iloc[idx] is not None:
            num_objs += 1
            if self.csv['x3'].iloc[idx] is not None:
                num_objs += 1

        nodule_data = self.csv.iloc[idx]
        boxes = []
        if nodule_data['label'].any()==1: # nodule data
            for i in range(num_objs):
                if i == 0 : 
                    x_min = int(nodule_data['x'])
                    y_min = int(nodule_data['y'])
                    y_max = int(y_min+nodule_data['height'])
                    x_max = int(x_min+nodule_data['width'])

                    boxes.append([x_min, y_min, x_max, y_max])
                else :
                    x_min = int(nodule_data[f'x{i+1}'])
                    y_min = int(nodule_data[f'y{i+1}'])
                    y_max = int(y_min+nodule_data[f'height{i+1}'])
                    x_max = int(x_min+nodule_data[f'width{i+1}'])

                    boxes.append([x_min, y_min, x_max, y_max])
            
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # for non-nodule images
        else:
            boxes = torch.empty([0,4], dtype=torch.float32)
            area = torch.tensor([0])
            labels = torch.zeros(1, dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        return boxes, labels, area, iscrowd

    def _process_mosaic(self,idx): # from yolov5
        labels4 = []
        boxes4 = []
        s = self.cfg.new_shape
        h, w = s, s
        mosaic_border = [-s // 2, -s // 2]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
        indices = [idx] + random.choices(range(self.__len__()), k=3) 

        random.shuffle(indices)
        for i, idx in enumerate(indices):
            img = self._process(idx)
            boxes, labels, _ , _ = self._process_labels(idx)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2), 0.5, dtype=np.float32)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)



            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # drop = []

            boxes_padded = boxes.clone()
            for i in range(len(boxes_padded)):
                boxes_padded[i,0] = boxes_padded[i,0] + padw
                boxes_padded[i,2] = boxes_padded[i,2] + padw
                boxes_padded[i,1] = boxes_padded[i,1] + padh
                boxes_padded[i,3] = boxes_padded[i,3] + padh

            boxes4.append(boxes_padded)
        drop = []
        boxes4 = torch.cat(boxes4, 0)


        for i, box in enumerate(boxes4):
            boxes4[i] = np.clip(box, 0, 2*s)
            if boxes4[i][0] == boxes4[i][2] or boxes4[i][1] == boxes4[i][3]:
                drop.append(i)

        if drop: # drop boxes with no area
            mask = torch.ones(boxes4.shape, dtype=torch.bool)
            mask[drop] = False
            num_boxes = boxes4.shape[0] - len(drop)
            boxes4 = boxes4[mask].reshape([num_boxes,4])
            
        labels4 = torch.tensor(len(boxes4)* [1]) # adjust number of positive labels

        return img4, boxes4, labels4 

    def __getitem__(self, idx):
        
        imgid = self.csv['img_name'].iloc[idx].replace('.mha','')
        boxes, labels, area, iscrowd = self._process_labels(idx)
        p = random.random()
        if self.preload:
            if not self.use_cache :
                if self.cfg.get('mosaic',0) > p and self.set == 'train':
                    img, labels  = self._process_mosaic(idx)
                else: 
                    img = self._process(idx)
                self.preLoadList[idx] = img 
            img = self.preLoadList[idx]
        else: 
            
            if self.cfg.get('mosaic',0) > p  and self.set == 'train':
                img, boxes, labels  = self._process_mosaic(idx)
            else:
                img = self._process(idx)

        target = {}
        target["labels"] = labels
        target["area"] = area
        target['image_id'] = imgid
        target["iscrowd"] = iscrowd
        target['set'] = self.set
        target['img_size'] = (self.cfg.new_shape, self.cfg.new_shape)
        target['img_scale'] = torch.tensor([1.0])


        if len(boxes)==0:
            lab = [] 
        else: 
            lab = labels.tolist()
        try:
            transformed = self.transform(image=img,bboxes=boxes,class_labels=lab) # TODO new_shape ..
        except:
            print('error in transformation')

        img = transformed['image'] 
        if len(transformed['bboxes']) == 0:
            target["boxes"] = torch.empty([0,4], dtype=torch.float32)
        else:
            target["boxes"] = torch.tensor(transformed['bboxes'])
        if not transformed['class_labels']:
            target['labels'] = torch.tensor([0])
        else :
            target['labels'] = torch.tensor(transformed['class_labels'])
        # img = transforms.ToTensor()(img)
        return img, target      

def get_transforms(cfg, set):
    ## transformations
        transforms = []
        if set == 'train': 
            if cfg.new_shape != 1024 or cfg.get('mosaic',False):
                transforms.append(A.Resize(cfg.new_shape,cfg.new_shape)) #,scale=(0.95,1),ratio=(0.95,1)))
            if cfg.get('randomCrop',False):
                transforms.append(A.RandomSizedBBoxSafeCrop(cfg.new_shape,cfg.new_shape)) #,scale=(0.95,1),ratio=(0.95,1)))
            if cfg.get('affine',False):
                transforms.append(A.Affine (scale=1.1, translate_percent=.1, translate_px=None, rotate=10, shear=10, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=0.5) ) 
            if cfg.get('colorJitter',False):
                transforms.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)) 
            if cfg.get('cutout',False):
                transforms.append(A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5)) 
            transforms.append(A.HorizontalFlip()) 
        if cfg.get('vinDR_augment',False): # similar augmentation that is used by the VinDR challenge
            transforms = [
                # A.RandomSizedBBoxSafeCrop(cfg.new_shape,cfg.new_shape),
                A.CropAndPad(px=[-50,50],p=0.5), # A.CropAndPad(px=[-50,50],p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2,
                    p=0.9
                ),


                A.HorizontalFlip(p=0.5),


                A.Rotate(
                    limit=5,
                    p=0.6,
                ),


                A.OneOf([
                    A.Blur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0)
                    ],
                    p=0.5),

                
                A.Resize(
                    height=cfg.new_shape,
                    width=cfg.new_shape,
                    p=1.0),
                    
                A.Cutout(
                    num_holes=int(0.05 * cfg.new_shape*cfg.new_shape / (cfg.new_shape//20)**2),
                    max_h_size=cfg.new_shape//20,
                    max_w_size=cfg.new_shape//20,
                    fill_value=0,
                    p=0.5),
                    
            ]

        if set == "val" or set == "test": 

            if cfg.new_shape != 1024:
                transforms.append(A.Resize(cfg.new_shape,cfg.new_shape)) 
            if cfg.get('randomResizeCrop',False):
                transforms.append(A.CenterCrop(cfg.new_shape, cfg.new_shape))
            if cfg.get('vinDR_augment',False):
                transforms = []
                if cfg.new_shape != 1024:
                    transforms.append(A.Resize(
                    height=cfg.new_shape,
                    width=cfg.new_shape,
                    p=1.0))

        transforms.append(ToTensorV2()) 
        return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])) 

