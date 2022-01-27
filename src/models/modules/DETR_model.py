import torchvision.models as models
from torch import nn
import torch
from src.models.modules.DETR.util.get_args import get_args_parser
from src.models.modules.DETR import build_model
import argparse
from src.utils.FrozenBN import FrozenBatchNorm2d
# from src.models.modules.conv2dsame import Conv2dSame

import torch.nn.functional as F 

class DETR(nn.Module): # 
    def __init__(self,cfg):
        super(DETR, self).__init__()

        num_classes = cfg.num_classes
        args = get_args_parser(cfg).parse_known_args()[0] # this is a mess but seems to work

        self.model, self.criterion, self.postprocessors = build_model(args)
        if cfg.pretrained:
            # print('loading weights')
            state_dict = torch.hub.load_state_dict_from_url(
                url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
                map_location='cpu',
                check_hash=True)
            del state_dict["model"]["class_embed.weight"]
            del state_dict["model"]["class_embed.bias"]
            del state_dict["model"]["query_embed.weight"]
            if cfg.get('reset_BN',False): # reset Batch norm parameters to initial values
                for m in state_dict['model']:
                    if 'bn' in m : # there are layer norm layers that are not reset or frozen atm
                        if 'bias' in m: 
                            state_dict['model'][m] = torch.zeros_like(state_dict['model'][m])
                        elif 'weight' in m:
                             state_dict['model'][m] = torch.ones_like(state_dict['model'][m])
                        elif 'running_mean' in m:
                            state_dict['model'][m] = torch.zeros_like(state_dict['model'][m])
                        elif 'running_var' in m:
                            state_dict['model'][m] = torch.zeros_like(state_dict['model'][m])
                        # else:
                        #     print('NI')
            self.model.load_state_dict(state_dict["model"], strict=False)
            if cfg.get('freeze_BN',False):
                    # self.model.apply(self.reset_batchnorm) # Reset BN parameters  
                FrozenBatchNorm2d.convert_frozen_batchnorm(self.model)


    def get_model(self):
        return self.model
    def get_criterion(self):
        return self.criterion
    def get_postprocessors(self):
        return self.postprocessors
    def reset_batchnorm(self,m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, FrozenBatchNorm2d):
            print('resetting', m)
            m.reset_parameters()
            # m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()