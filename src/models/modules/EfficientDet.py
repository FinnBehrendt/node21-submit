import torchvision.models as models
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, create_model, load_pretrained, create_dataset
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
from torch import nn
import torch
import timm
from src.utils.FrozenBN import FrozenBatchNorm2d
from typing import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from timm.models.layers.conv2d_same import Conv2dSame
# from src.models.modules.conv2dsame import Conv2dSame

import torch.nn.functional as F 

class EffDet(nn.Module): # 
    def __init__(self,cfg):
        super(EffDet, self).__init__()

        architecture=cfg.version
        config = get_efficientdet_config(architecture)
        # config.update({'num_classes': cfg.num_classes})
        config.update({'image_size': (cfg.new_shape, cfg.new_shape)})
        config.update({'max_det_per_image':100})
        config.update({'extra_loss_weight':6.0}) 
        if cfg.get('label_smoothing',0):
            config.update({'label_smoothing':cfg.get('label_smoothing',0)})
        if cfg.get('focal_loss',False):
            config.update({'legacy_focal':True})
        # config.update({'norm_layer':nn.BatchNorm2d})
        # config.update({'create_labeler': False})
        # config.update({'url':'https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/efficientdet_d1-bb7e98fe.pth'})
        # print(config)



        model = EfficientDet(config, pretrained_backbone=False)

        if cfg.pretrained:
            load_pretrained(model, config.url)

        if cfg.get('vindr_weights',False):
            
            try:
                state_dict = torch.load(cfg.get('vindr_path','/opt/algorithm/fastercnn50.pth'),map_location=torch.device('cpu'))['model_state_dict']
                new_statedict = OrderedDict()
                for key_load,key in zip(state_dict,model.state_dict()): 
                    new_statedict[key] = state_dict[key_load]
                new_statedict['backbone.conv_stem.weight'] = new_statedict['backbone.conv_stem.weight'][:,0:3]
                del new_statedict["class_net.predict.conv_pw.weight"]
                del new_statedict["class_net.predict.conv_pw.bias"]
                model.load_state_dict(new_statedict,strict=False)
            except:
                print('FAILED TO LOAD STATE DICT - THIS IS OK IN EVALUATION')
        if cfg.num_classes is not None and cfg.num_classes != config.num_classes:
            model.reset_head(num_classes=cfg.num_classes)
        if cfg.get('freeze_BN',False):
            FrozenBatchNorm2d.convert_frozen_batchnorm(model)

        self.model = DetBenchTrain(model,config) # could use DetBenchpredict for evaluation in the future 
        

    def get_model(self):
        return self.model
