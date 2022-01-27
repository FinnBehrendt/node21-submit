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
import os
from efficientnet_pytorch import EfficientNet
from torchvision.ops.misc import FrozenBatchNorm2d
import wandb
import matplotlib.pyplot as plt
import time
from src.utils.utils import intersection_over_union, get_NonMaxSup_boxes
from src.utils.custom_metrics import calc_iou, calc_FROC
from monai.metrics import compute_froc_score
from src.utils.utils import xyxy2yxyx, yxyx2xyxy
from src.utils.scheduler import scheduler_lambda

from torchvision import transforms
from src.utils.utils import get_NonMaxSup_boxes
from src.utils.FrozenBN import FrozenBatchNorm2d
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
        self.prediction_confidence_threshold = cfg.prediction_confidence_threshold
        if cfg.name == 'EfficientDet' : 
            from src.models.modules.EfficientDet import EffDet
            self.modelclass = EffDet(cfg)
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

    def forward(self, images, targets): # unused here..
        x = self.model(images,targets)
        return x

    def _calc_train_loss(self, batch):
        images, annotations,  targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        images = torch.cat([images,images,images],1) # to rgb
        loss_dict = self.model(images,annotations)

        return loss_dict

    def _calc_val_loss(self, batch): # only input image for validation
        images, annotations,  targets = batch
        images = torch.cat([images,images,images],1) # to rgb
        loss_dict = self.model(images,annotations)

        return loss_dict


    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch == 0:
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
        detections = preds['detections'] 
        loss = preds['loss']
        loss_dict = {'box_loss': preds['box_loss'], 'class_loss':preds['class_loss'] }
        preds_list = []
        for i in range(detections.shape[0]):
            postprocessed = self._postprocess_single_prediction_detections(detections[i]) #return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
            boxes = postprocessed['boxes']
            scores = postprocessed['scores']
            if len(scores) == 0: 
                scores = torch.tensor([0.0],device=self.device)
            preds_list.append({'boxes':boxes , 'scores':scores })

        iou = [calc_iou(o, t) for t, o in zip(preds_list, batch[2])]
        self.log(f'{self.prefix}val/losses/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {'preds': preds_list, 'targets': batch[2], 'iou': iou,"loss": loss, "log": loss_dict}

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
        detections = preds['detections'] 
        preds_list = []
        for i in range(detections.shape[0]):
            postprocessed = self._postprocess_single_prediction_detections(detections[i]) #return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
            boxes = postprocessed['boxes']
            scores = postprocessed['scores']
            if len(scores) == 0: 
                scores = torch.tensor([0.0],device=self.device)
            preds_list.append({'boxes':boxes , 'scores':scores })

        iou = [calc_iou(o, t) for t, o in zip(preds_list, batch[2])]
        return {'preds': preds_list, 'targets': batch[2], 'iou': iou}

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

        detections = preds['detections'] 
        preds_list = []
        for i in range(detections.shape[0]):
            postprocessed = self._postprocess_single_prediction_detections(detections[i]) #return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
            boxes = postprocessed['boxes']
            scores = postprocessed['scores']
            if len(scores) == 0: 
                scores = torch.tensor([0.0],device=self.device)
            preds_list.append({'boxes':boxes , 'scores':scores })
        for i, pred in enumerate(preds_list):
            preds_list[i] = get_NonMaxSup_boxes(preds_list[i])
        return {"preds": preds_list, "targets": batch[2], "images": batch[0]}

    def configure_optimizers(self):
        # Optimizer:
        return_dict = {}
        # lr_scaled = self.lr * self.trainer.num_gpus * self.bs / 64
        if self.optim == 'Adam' :
            if self.cfg.get('custom_optim',False):
                params_v  = sum([

                    list( self.model.model.backbone.conv_stem.parameters() ),
                    list( self.model.model.backbone.bn1.parameters() ),

                    list( self.model.model.fpn.parameters() ),
                    list( self.model.model.class_net.parameters() ),
                    list( self.model.model.box_net.parameters() )
                    # list( self.backbone.extra_net.parameters() ) if use_extra_net else [],
                ], [])
                
                param_id_v = [id(p) for p in params_v]
                if self.cfg.get('weight_decay',0) > 0.0:
                    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
                    for k, v in self.named_modules():
                        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                            if id(v.bias) in param_id_v:
                                pg2.append(v.bias)  # biases, no decay
                            else:
                                v.bias.requires_grad_(False)
                        
                        if isinstance(v, nn.BatchNorm2d) or isinstance(v,FrozenBatchNorm2d):
                            if id(v.weight) in param_id_v:
                                pg0.append(v.weight)  # weights, no decay
                            else:
                                v.weight.requires_grad_(False)
                            
                        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                            if id(v.weight) in param_id_v:
                                pg1.append(v.weight)  # weights, decay
                            else:
                                v.weight.requires_grad_(False)

                    if not self.cfg.get('freeze_BN',False):
                        # Weights Nodecay
                        return_dict['optimizer'] = optim.Adam(pg0, lr=self.lr)
                        
                        # Weights Decay
                        return_dict['optimizer'].add_param_group({'params': pg1, 'weight_decay': self.cfg.get('weight_decay',0)})
                        
                        # Biasese NoDecay
                        return_dict['optimizer'].add_param_group({'params': pg2})
                    else:  # we dont update BN params here
                        # Weights Decay
                        return_dict['optimizer'] = optim.Adam(pg1, lr=self.lr)
                        
                        # Biasese NoDecay
                        return_dict['optimizer'].add_param_group({'params': pg2})

                    del pg0, pg1, pg2
                else:   
                    return_dict['optimizer'] = optim.Adam(
                        params_v,
                        lr=self.lr,
                        weight_decay=self.cfg.get('weight_decay',0),
                    )
                    
                n_w = 0
                print('Optimizable parameters:')
                for i_p, (n, p) in enumerate( list( self.model.model.named_parameters() )):
                    if p.requires_grad:
                        print('{:4d}  {:s}  {:50s}  {:}'.format(i_p, 'OPT' if p.requires_grad else '---',  n, p.shape) )
                        n_w += np.prod(p.shape)
                    else:
                        print('{:4d}  {:s}  {:50s}  {:}'.format(i_p, 'OPT' if p.requires_grad else '---',  n, p.shape) )
                        
                print(f'Total optimizable weights: {n_w/1e6:0.02f} Mw')
            else: 
                return_dict['optimizer'] = optim.Adam(self.parameters(), lr = self.lr,weight_decay=self.cfg.get('weight_decay',0))
        elif self.optim == 'AdamW' :
            return_dict['optimizer'] = optim.AdamW(self.parameters(), lr = self.lr )
        elif self.optim == 'SGD':
            return_dict['optimizer'] = optim.SGD(self.parameters(), lr = self.lr, momentum=0.9, weight_decay=0.0005 )
        elif self.optim == 'Lamb' :
            return_dict['optimizer'] = optim_third_party.Lamb(self.parameters(), lr = self.lr)
        else:
            raise NotImplementedError("Optimizer is not implemented")
        
        # LR Schedule
        if self.LRSched is not None: # meaningless if based on epochs...
            if self.LRSched=='plateau' :
                return_dict['lr_scheduler'] = { 'scheduler' : torch.optim.lr_scheduler.ReduceLROnPlateau(return_dict['optimizer'], factor = self.cfg.get('sched_factor',0.5), patience = self.cfg.get('sched_patience',4), verbose =True  ),
                                                'monitor': f'{self.prefix}val/Competition_metric/' }
            if self.LRSched=='step' :
                return_dict['lr_scheduler'] = { 'scheduler' : torch.optim.lr_scheduler.StepLR(return_dict['optimizer'], gamma = self.cfg.get('sched_factor',0.5), step_size = self.cfg.get('sched_patience',4),verbose =True ),
                                                }
            if self.LRSched=='custom' :
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
        #     if self.warmup  :
        #         return_dict['lr_scheduler'] = { 'scheduler' : torch.optim.lr_scheduler.LinearLR(return_dict['optimizer'], start_factor =0.00001,end_factor =1.0 ,total_iters =1000/self.bs,verbose=False), 'interval':'step', 'frequency':1
        #                                         }                                            
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


    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        indexes = torch.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

    