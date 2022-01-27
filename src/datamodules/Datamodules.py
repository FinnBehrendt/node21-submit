from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
import src.datamodules.get_data as get_data
from src.utils.utils import collate_fn
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
from src.utils.utils import xyxy2yxyx, yxyx2xyxy
from torch.utils.data import WeightedRandomSampler
class Nodule21(LightningDataModule):  # data module for CheXpert Competition Task - 5 pathologies
    def __init__(self, cfg, fold = None, transform=None, data_aug=None):
        super(Nodule21, self).__init__()
        self.cfg = cfg
        # self.num_gpus = 1 # TODO
        self.sample_set = cfg.get('sample_set',False) #  for debugging
        if cfg.get('effdet',False):
            self.collate_fn = self.collate_fn_effdet
        else: 
            self.collate_fn = collate_fn
        # specify paths
        self.imgpath = {}
        self.csvpath_train = cfg.path.train.labels[fold]
        self.imgpath['train'] = cfg.path.train.images
        self.csvpath_val = cfg.path.val.labels[fold]
        self.imgpath['val'] = cfg.path.val.images
        # if cfg.get('all_sets_mixed',False) and not cfg.get('use_generated_nodules',False) and not cfg.get('final_set',False):
        #     print('Using Train/val/test split Version 3')
        #     self.csvpath_train =  self.csvpath_train.replace('.csv','_v3.csv')
        #     self.csvpath_val = self.csvpath_val.replace('.csv','_v3.csv')
        # if cfg.get('use_generated_nodules',False) and not cfg.get('final_set',False):
        #     print('Using Train/val/ split Version 4 with generated nodules - set oversampling to false!')
        #     self.csvpath_train =  self.csvpath_train.replace('.csv','_v4gen.csv')
        #     self.csvpath_val = self.csvpath_val.replace('.csv','_v4gen.csv')
        #     self.cfg.imbalancedSampling=False 
        #     self.cfg.imbalancedSamplingTest = False
        # if cfg.get('final_set',False):
        #     print('using all data for final run without testset')
        #     self.csvpath_train =  self.csvpath_train.replace('.csv','_finalgen.csv')
        #     self.csvpath_val = self.csvpath_val.replace('.csv','_finalgen.csv')
        #     self.cfg.imbalancedSampling=False 
        #     self.cfg.imbalancedSamplingTest = False
        # read csv as pandas dataframe
        self.labels = {'train': [],
                        'val': [],
                        'test': []}
        self.csv = {}
        states = ['train','val']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)

        if cfg.test : # only for final testing
            states.append('test')
            self.csvpath_test = cfg.path.test.labels
            if cfg.get('all_sets_mixed',False):
                self.csvpath_test = self.csvpath_test.replace('.csv','_v3.csv')
            self.imgpath['test'] = cfg.path.test.images
            self.csv['test'] = pd.read_csv(self.csvpath_test)

        for state in states:
            self.csv[state]['x2'] = None
            self.csv[state]['y2'] = None
            self.csv[state]['x3'] = None
            self.csv[state]['y3'] = None
            self.csv[state]['width2'] = None
            self.csv[state]['width3'] = None
            self.csv[state]['height2'] = None
            self.csv[state]['height3'] = None
            duplicates = self.csv[state][self.csv[state].img_name.duplicated(keep=False)]
            unique_list = duplicates.img_name.unique()
            for pat in unique_list: 
                xs = []
                ys = []
                widths = []
                heights = []
                for i, (idx, ann) in enumerate(duplicates[duplicates.img_name == pat].iterrows()): 

                    if i == 0: 
                        keep_id = idx
                    else:
                        self.csv[state] = self.csv[state].drop(index=idx)
                        widths.append(ann.width)
                        heights.append(ann.height)
                        xs.append(ann.x)
                        ys.append(ann.y)
                        
                self.csv[state].x2.loc[keep_id] = xs[0]
                self.csv[state].y2.loc[keep_id] = ys[0]
                self.csv[state].width2.loc[keep_id] = widths[0]
                self.csv[state].height2.loc[keep_id] = heights[0]
                if len(xs) > 1:
                    self.csv[state].x3.loc[keep_id] = xs[1]
                    self.csv[state].y3.loc[keep_id] = ys[1]
                    self.csv[state].width3.loc[keep_id] = widths[1]
                    self.csv[state].height3.loc[keep_id] = heights[1]
        #     patientid = self.csv[state].Path.str.split("train/", expand=True)[1]
            self.csv[state]['set']=state
           

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if self.sample_set: # debug with small training set
            self.train = get_data.DataSet(self.csv['train'].sample(frac=.1), self.cfg)
            self.val = get_data.DataSet(self.csv['val'].sample(frac=.1), self.cfg)
            # self.test = get_data.DataSet(self.csv['test'].sample(frac=.1), self.cfg)
            # self.test = get_data.DataSet(self.csv['test'][self.csv['test'].label==1].append(self.csv['test'][self.csv['test'].label==0][0:227]), self.cfg)

        else: 
            if self.cfg.get('alldata_noCV',False):
                print('Using all data at once without Cross Validation')
                self.train = get_data.DataSet(self.csv['train'].append(self.csv['val']), self.cfg)
                self.val = get_data.DataSet(self.csv['test'], self.cfg)
                # self.test = get_data.DataSet(self.csv['test'], self.cfg)

            else:
                self.train = get_data.DataSet(self.csv['train'], self.cfg)
                self.val = get_data.DataSet(self.csv['val'], self.cfg)
                # self.test = get_data.DataSet(self.csv['test'], self.cfg)
            
            # self.test = get_data.DataSet(self.csv['test'][self.csv['test'].label==1][0:93].append(self.csv['test'][self.csv['test'].label==0]), self.cfg)


    def train_dataloader(self):

        if self.cfg.get('imbalancedSampling',False):
            shuffle = False
            if self.cfg.get('undersampling',False): 
                sampler = ImbalancedDatasetSampler(self.train)
            else:
                labels = self.train.get_labels()
                class_count = np.unique(labels, return_counts=True)[1]
                weight = 1. / class_count
                # weight = np.array([1,class_count[0]/class_count[1]])
                samples_weight = weight[labels]
                samples_weight = torch.from_numpy(samples_weight)
                sampler = WeightedRandomSampler(samples_weight, int(class_count[0]*2),replacement=self.cfg.get('replacement',True))
        else:
            sampler= None
            shuffle = True
        return DataLoader(self.train,sampler=sampler, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers * int(self.trainer.num_gpus), pin_memory=True, shuffle=shuffle, collate_fn=self.collate_fn, drop_last=True)

    def val_dataloader(self):
        if self.cfg.get('imbalancedSamplingTest',False):

            shuffle = False
            labels = self.val.get_labels()
            class_count = np.unique(labels, return_counts=True)[1]
            weight = 1. / class_count
            # weight = np.array([1,class_count[0]/class_count[1]])
            samples_weight = weight[labels]
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight,int(class_count[1]*2),replacement=False)
        else:
            sampler= None
            shuffle = True
        return DataLoader(self.val,sampler=sampler, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers *  int(self.trainer.num_gpus), pin_memory=True, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        # if self.cfg.get('imbalancedSamplingTest',False):

        #     shuffle = False
        #     labels = self.test.get_labels()
        #     class_count = np.unique(labels, return_counts=True)[1]
        #     weight = 1. / class_count
        #     # weight = np.array([1,class_count[0]/class_count[1]])
        #     samples_weight = weight[labels]
        #     samples_weight = torch.from_numpy(samples_weight)
        #     sampler = WeightedRandomSampler(samples_weight, int(class_count[0]*2),replacement=False)
        # else:
        sampler= None

        if self.trainer:
            num_gpus = int(self.trainer.num_gpus)
        else: num_gpus = 1

        return DataLoader(self.test,sampler=sampler, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers * num_gpus, pin_memory=True, shuffle=False, collate_fn=self.collate_fn)
    
    @staticmethod
    def collate_fn_effdet(batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        # annotations should be in yxyx format for effdet
        boxes_xyxy =  xyxy2yxyx(deepcopy(targets))
        boxes = [target["boxes"].float() for target in boxes_xyxy ]
        labels = [target["labels"].float() for target in targets]
        for i, label in enumerate(labels):
            if len(label) < 2 and label == 0 : 
                labels[i] = torch.tensor([])
        # labels = [torch.tensor([]) for x in labels if x == 0 ]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets

