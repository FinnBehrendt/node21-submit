import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
import torch.optim as optim
import torch_optimizer as optim_third_party
from torchmetrics import MetricCollection, AUROC, MAP
from typing import Any, List
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from pathlib import Path
import pickle
import sys, os
import wandb
import matplotlib.pyplot as plt
import time
from src.utils.utils import intersection_over_union, get_NonMaxSup_boxes
from src.utils.custom_metrics import calc_iou, calc_FROC
from monai.metrics import compute_froc_score
from src.utils.scheduler import scheduler_lambda
 
class Detector(LightningModule): 
    ''' Class for the Detection tasks with different (pretrained) models '''
    def __init__(self,cfg,prefix=None):
        super().__init__()

        self.cfg = cfg
        self.multiCrop = cfg.multiCrop 
        self.thres = cfg.decision_thres
        self.lr = cfg.lr
        self.bs = cfg.batch_size
        self.combineCrops = cfg.combineCrops
        self.cropAttention = cfg.cropAttention

        if cfg.name == 'FasterCNN' : 
            from src.models.modules.FasterCNN import FasterCNN
            self.modelclass = FasterCNN(cfg)
            self.model = self.modelclass.get_model()

        if cfg.name == 'RetinaNet' : 
            from src.models.modules.RetinaNet import RetinaNet
            self.modelclass = RetinaNet(cfg)
            self.model = self.modelclass.get_model()

        # Optimizer
        self.optim = cfg.optim
        # LR Schedule
        self.LRSched = cfg.get('LR_Scheduler',None)
        self.warmup = cfg.get('LR_warmup',False)
        self.warmup_steps = cfg.get('warmup_steps',1000) / self.bs
        # convenient for n-Fold CV
        self.prefix = prefix
        
        self.save_hyperparameters()

    def forward(self, x): # unused here..
        x = self.model(x)
        return x

    def _calc_train_loss(self, batch):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images,targets)

        return loss_dict

    def _calc_val_loss(self, batch): # only input image for validation
        images, _ = batch
        images = list(image for image in images)
        loss_dict = self.model(images)

        return loss_dict


    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch == 0: # preloading 
            self.trainer.datamodule.train.set_use_cache(True)
        loss_dict = self._calc_train_loss(batch)
        loss = sum(loss for loss in loss_dict.values())
        # log train metrics
        self.log(f'{self.prefix}train/losses/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "log": loss_dict}

    def training_epoch_end(self,outputs) -> None:
        if self.trainer.current_epoch == 0:
            self.trainer.datamodule.train.set_use_cache(True)

    def validation_step(self, batch, batch_idx) :
        preds = self._calc_val_loss(batch)
        with torch.no_grad(): # calculate the validation loss --> Model has to be in training mode
            self.model.train()
            loss_dict = self._calc_train_loss(batch)
            loss = sum(loss for loss in loss_dict.values())
            self.model.eval() # Need to get back to eval mode

        for sample in preds: 
            if len(sample['scores']) == 0: 
                sample['scores'] = torch.tensor([0.0],device=self.device)
        iou = [calc_iou(o, t) for t, o in zip(preds, batch[1])]
        self.log(f'{self.prefix}val/losses/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'preds': preds, 'targets': batch[1], 'iou': iou,"loss": loss, "log": loss_dict}

    def validation_epoch_end(self, outputs) -> None:
        if self.trainer.current_epoch == 0:
            self.trainer.datamodule.val.set_use_cache(True)
        # Rearranging dicts
        max_preds = []
        target_scores = []
        boxes_target = []
        boxes_pred = []
        batches_pred = [o['preds'] for o in outputs]
        batches_target = [o['targets'] for o in outputs]
        [boxes_target.extend(target) for target in batches_target]
        [boxes_pred.extend(pred) for pred in batches_pred]

        for batch in outputs:
            for preds,targets in zip(batch['preds'],batch['targets']) : # preds and targets
                max_preds.append(preds['scores'].max())
                target_scores.append(targets['labels'].max())

        metric_collection_classification = MetricCollection(AUROC())
        metric_class = metric_collection_classification(torch.stack(max_preds),torch.stack(target_scores))
        batch_iou = [o['iou'] for o  in outputs]
        avg_iou =  torch.cat([torch.tensor(x) for x in batch_iou]).nanmean()
        
        fps, sens = calc_FROC(boxes_pred, boxes_target)
        froc_0125 = compute_froc_score(fps,sens,eval_thresholds = (0.125))
        froc_025 = compute_froc_score(fps,sens,eval_thresholds = (0.25))
        froc_05 = compute_froc_score(fps,sens,eval_thresholds = (0.5))
        froc_all = compute_froc_score(fps,sens,eval_thresholds = (0.125,0.25,0.5))

        for key in metric_class:
            if key == 'AUROC': 
                pbar = True
            else: 
                pbar = False
            self.log(f'{self.prefix}val/{key}/', metric_class[key], sync_dist=True,  prog_bar=pbar)
        self.log(f'{self.prefix}val/IoU/', avg_iou, sync_dist=True)
        self.log(f'{self.prefix}val/froc_0125/', froc_0125, sync_dist=True)
        self.log(f'{self.prefix}val/froc_025/', froc_025, sync_dist=True)
        self.log(f'{self.prefix}val/froc_05/', froc_05, sync_dist=True)
        self.log(f'{self.prefix}val/froc_all/', froc_all, sync_dist=True)
        self.log(f'{self.prefix}val/Competition_metric/',0.75 * metric_class['AUROC'] + 0.25 * froc_025, sync_dist=True)
        
    def test_step(self, batch, batch_idx) :
        preds = self._calc_val_loss(batch)
        for sample in preds: 
            if len(sample['scores']) == 0: 
                sample['scores'] = torch.tensor([0.0],device=self.device)
        iou = [calc_iou(o, t) for t, o in zip(preds, batch[1])]
        return {'preds': preds, 'targets': batch[1], 'iou': iou}

    def test_epoch_end(self, outputs) -> None:

        # Rearranging dicts
        max_preds = []
        target_scores = []
        boxes_target = []
        boxes_pred = []
        batches_pred = [o['preds'] for o in outputs]
        batches_target = [o['targets'] for o in outputs]
        [boxes_target.extend(target) for target in batches_target]
        [boxes_pred.extend(pred) for pred in batches_pred]

        for batch in outputs:
            for preds,targets in zip(batch['preds'],batch['targets']) : # preds and targets
                max_preds.append(preds['scores'].max())
                target_scores.append(targets['labels'].max())

        metric_collection_classification = MetricCollection(AUROC())
        metric_class = metric_collection_classification(torch.stack(max_preds),torch.stack(target_scores))

        batch_iou = [o['iou'] for o  in outputs]
        avg_iou =  torch.cat([torch.tensor(x) for x in batch_iou]).nanmean()
        fps, sens = calc_FROC(boxes_pred, boxes_target)
        froc_0125 = compute_froc_score(fps,sens,eval_thresholds = (0.125))
        froc_025 = compute_froc_score(fps,sens,eval_thresholds = (0.25))
        froc_05 = compute_froc_score(fps,sens,eval_thresholds = (0.5))
        froc_all = compute_froc_score(fps,sens,eval_thresholds = (0.125,0.25,0.5))
        
        for key in metric_class:
            if key == 'AUROC': 
                pbar = True
            else: 
                pbar = False
            self.log(f'{self.prefix}test/{key}/', metric_class[key], sync_dist=True,  prog_bar=pbar)
        self.log(f'{self.prefix}test/IoU/', avg_iou, sync_dist=True)
        self.log(f'{self.prefix}test/froc_0125/', froc_0125, sync_dist=True)
        self.log(f'{self.prefix}test/froc_025/', froc_025, sync_dist=True)
        self.log(f'{self.prefix}test/froc_05/', froc_05, sync_dist=True)
        self.log(f'{self.prefix}test/froc_all/', froc_all, sync_dist=True)
        self.log(f'{self.prefix}test/Competition_metric/',0.75 * metric_class['AUROC'] + 0.25 * froc_025, sync_dist=True)

    def predict_step(self, batch, batch_idx) :
        preds = self._calc_val_loss(batch)
        for i, pred in enumerate(preds):
            preds[i] = get_NonMaxSup_boxes(preds[i])
        return {"preds": preds, "targets": batch[1], "images": batch[0]}


    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first n steps -> this is handled in the optimizer. 
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) /self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.cfg.lr
        
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        # Optimizer:
        return_dict = {}
        # lr_scaled = self.lr * self.trainer.num_gpus * self.bs / 64 # if multi GPU
        if self.optim == 'Adam' :
            return_dict['optimizer'] = optim.Adam(self.parameters(), lr = self.lr  )
        elif self.optim == 'AdamW' :
            return_dict['optimizer'] = optim.AdamW(self.parameters(), lr = self.lr )
        elif self.optim == 'SGD':
            return_dict['optimizer'] = optim.SGD(self.parameters(), lr = self.lr, momentum=0.9, weight_decay=0.0005 )
        elif self.optim == 'Lamb' :
            return_dict['optimizer'] = optim_third_party.Lamb(self.parameters(), lr = self.lr)
        else:
            raise NotImplementedError("Optimizer is not implemented")
        
        # LR Schedule
        if self.LRSched is not None: 
            if self.LRSched=='plateau' :
                return_dict['lr_scheduler'] = { 'scheduler' : torch.optim.lr_scheduler.ReduceLROnPlateau(return_dict['optimizer'], factor = self.cfg.get('sched_factor',0.5), patience = self.cfg.get('sched_patience',4), verbose =True  ),
                                                'monitor': f'{self.prefix}val/Competition_metric/' }
            if self.LRSched=='step' :
                return_dict['lr_scheduler'] = { 'scheduler' : torch.optim.lr_scheduler.StepLR(return_dict['optimizer'], gamma = self.cfg.get('sched_factor',0.5), step_size = self.cfg.get('sched_patience',4),verbose =True ),
                                                }
            if self.LRSched=='custom' : # cosine annealing with warmup
                import torch.optim.lr_scheduler as lr_scheduler
                lr_lf = scheduler_lambda(
                                        lr_frac=1e-3,
                                        warmup_epochs=5,
                                        cos_decay_epochs=60)
                scheduler = lr_scheduler.LambdaLR(
                                        return_dict['optimizer'],
                                        lr_lambda=lr_lf)            
                return_dict['lr_scheduler'] = { 'scheduler' : scheduler
                                                }
                                            
            
        return return_dict

    def eval_ensemble(self,ouputs_ensemble):
        boxes_pred, boxes_target = ouputs_ensemble['test-preds'], ouputs_ensemble['test-targets']
        max_preds = []
        target_scores = []

        for preds,targets in zip(ouputs_ensemble['test-preds'],ouputs_ensemble['test-targets']) : # preds and targets
            if len(preds['scores']) == 0: 
                max_preds.append(torch.tensor(0.0,dtype=torch.float64,device=self.device))
            else:
                max_preds.append(preds['scores'].max())
            target_scores.append(targets['labels'].max())

        metric_collection_classification = MetricCollection(AUROC())
        metric_class = metric_collection_classification(torch.stack(max_preds),torch.stack(target_scores))
        fps, sens = calc_FROC(boxes_pred, boxes_target)
        froc_0125 = compute_froc_score(fps,sens,eval_thresholds = (0.125))
        froc_025 = compute_froc_score(fps,sens,eval_thresholds = (0.25))
        froc_05 = compute_froc_score(fps,sens,eval_thresholds = (0.5))
        froc_all = compute_froc_score(fps,sens,eval_thresholds = (0.125,0.25,0.5))
        comp_metric = 0.75 * metric_class['AUROC'] + 0.25 * froc_025
        out_dict = {'FROC_all': froc_all, 'FROC_0125': froc_0125, 'FROC_025': froc_025, 'FROC_05': froc_05, 'AUROC':metric_class['AUROC'], 'Competition_metric': comp_metric }
        return out_dict, sens, fps

    