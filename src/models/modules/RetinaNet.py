import torchvision.models as models
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torch import nn
import torch
import math 
import torchvision
import torch.nn.functional as F 
from typing import OrderedDict
class RetinaNet(nn.Module): 
    def __init__(self,cfg):
        super(RetinaNet, self).__init__()
        if cfg.get('customBackbone',None):
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(cfg.customBackbone, pretrained = cfg.pretrained_backbone, returned_layers=[2, 3, 4],
                                   extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelP6P7(256, 256), trainable_layers=cfg.get('trainable_layers',5))
            self.model = torchvision.models.detection.RetinaNet(backbone=backbone, num_classes=2)
        elif cfg.get('custom_anchors',False):
            anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            aspect_ratios = ((0.8, 1.0, 1.2),) * len(anchor_sizes)
            anchor_generator = torchvision.models.detection.retinanet.AnchorGenerator(anchor_sizes,
                                    aspect_ratios)
            self.model = retinanet_resnet50_fpn(
                                pretrained = cfg.pretrained,
                                pretrained_backbone = cfg.pretrained_backbone,
                                trainable_backbone_layers = cfg.get('trainable_layers',5),
                                anchor_generator = anchor_generator
                                )
                                    
        else: 
            self.model = retinanet_resnet50_fpn(
                                            pretrained = cfg.pretrained,
                                            pretrained_backbone = cfg.pretrained_backbone,
                                            trainable_backbone_layers = cfg.get('trainable_layers',5)
                                            )



        # replace classification layer 
        in_features = self.model.head.classification_head.conv[0].in_channels
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.num_classes = cfg.num_classes

        cls_logits = torch.nn.Conv2d(in_features, num_anchors * cfg.num_classes, kernel_size = 3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
        # assign cls head to model
        self.model.head.classification_head.cls_logits = cls_logits

    def get_model(self):
        return self.model
