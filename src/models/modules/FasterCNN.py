from typing import OrderedDict
import torchvision.models as models
# from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
import torchvision
# from src.models.modules.torchvision_models.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_custom 
from torch import nn
import torch
import torchxrayvision as xrv
from src.utils.losses import FocalLoss
import torch.nn.functional as F 

class FasterCNN(nn.Module): # standard DenseNet121
    def __init__(self,cfg):
        super(FasterCNN, self).__init__()
        if cfg.get('label_smoothing',False):
            torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_ls
        elif cfg.get('class_weighting',False):
            torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_cw
        elif cfg.get('focal_loss',False):
            torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_focal
        elif cfg.get('class_weighting',False) and cfg.get('class_weighting',False) :
            torchvision.models.detection.roi_heads.fastrcnn_loss = fastercnn_loss_both # this is so bad...

        if cfg.get('xray_weights',False):
            anchor_generator = torchvision.models.detection.faster_rcnn.AnchorGenerator(sizes=((32, 64, 128, 256),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
            backbone_net = xrv.models.DenseNet(weights="densenet121-res224-all").features
            backbone_net.out_channels = 1024
            out_channels = 1024
            pretrained_weights = backbone_net[0].weight
            
            backbone_net[0] = nn.Conv2d(3, 64, kernel_size = (7,7), stride=(2,2), padding = (3,3), bias = False) # accept 3 channel input
            backbone_net[0].weight.data = torch.cat([pretrained_weights.data,pretrained_weights.data,pretrained_weights.data],1) # update weights of the first 2 channels from pretraining
            
            backbone = nn.Sequential(
            backbone_net)
            backbone.out_channels = 1024
            
            model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=2,rpn_anchor_generator=anchor_generator)
        
        elif cfg.get('Chexpert_weights',False): # pretrained with Chexpert classification task
            
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                                        pretrained = cfg.pretrained,
                                        pretrained_backbone = cfg.pretrained_backbone,
                                        trainable_backbone_layers = cfg.get('trainable_layers',5)
                                        )
            state_dict = torch.load('/home/linux/Node21/pretrained/FCRNN/epoch-8_step-20105_loss-4.17.ckpt')['state_dict']
            state_dict.pop('model.classifier.weight')
            state_dict.pop('model.classifier.bias')
            state_dict.pop('criterion.pos_weight')
            new_statedict = OrderedDict()
            for key_load,key in zip(state_dict,model.backbone.state_dict()): 
                new_statedict[key] = state_dict[key_load]

            model.backbone.load_state_dict(new_statedict)
            layers_to_train = ['body.layer4', 'body.layer3', 'body.layer2', 'body.layer1', 'body.conv1'][:cfg.get('trainable_layers',5)]
            if cfg.get('trainable_layers',5) == 5:
                layers_to_train.append('bn1')
            for name, parameter in model.backbone.named_parameters():
                if all([not name.startswith(layer) for layer in layers_to_train]) and not 'fpn' in name:
                    parameter.requires_grad_(False)
                    print(f'Freezing              {name}')

        elif cfg.get('customBackbone',None):
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(cfg.customBackbone,pretrained = cfg.pretrained_backbone, trainable_layers=cfg.get('trainable_layers',5))
            model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=2)
        
        elif cfg.get('custom_anchors',False):

            anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
            aspect_ratios = ((0.8, 1.0, 1.2),) * len(anchor_sizes)
            anchor_generator = torchvision.models.detection.faster_rcnn.AnchorGenerator(anchor_sizes,
                                    aspect_ratios)
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                                        # num_classes=num_classes,
                                        pretrained = cfg.pretrained,
                                        pretrained_backbone = cfg.pretrained_backbone,
                                        trainable_backbone_layers = cfg.get('trainable_layers',5),
                                        rpn_anchor_generator = anchor_generator

                                        )
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                                            # num_classes=num_classes,
                                            pretrained = cfg.pretrained,
                                            pretrained_backbone = cfg.pretrained_backbone,
                                            trainable_backbone_layers = cfg.get('trainable_layers',5)

                                            )


        in_features = model.roi_heads.box_predictor.cls_score.in_features
        head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, cfg.num_classes)
        model.roi_heads.box_predictor = head
        if cfg.get('vindr_weights',False) and not cfg.get('customBackbone',None):
            try:
                state_dict = torch.load(cfg.get('vindr_path','/opt/algorithm/fastercnn50.pth'),map_location=torch.device('cpu'))
                new_statedict = OrderedDict()
                for key_load,key in zip(state_dict,model.state_dict()): 
                    new_statedict[key] = state_dict[key_load]
                del new_statedict["roi_heads.box_predictor.cls_score.weight"]
                del new_statedict["roi_heads.box_predictor.cls_score.bias"]
                del new_statedict["roi_heads.box_predictor.bbox_pred.weight"]
                del new_statedict["roi_heads.box_predictor.bbox_pred.bias"]
                model.load_state_dict(new_statedict,strict=False)
            except:
                print('loading of ckpt failed. This is ok in Evaluation')
        self.model = model


    def get_model(self):
        return self.model

# label smoothing, class weighting 
custom_class_loss_ls = torch.nn.CrossEntropyLoss(label_smoothing=0.05, weight=None) 
custom_class_loss_cw = torch.nn.CrossEntropyLoss(label_smoothing=0, weight=torch.tensor([1,2.75])) 
custom_class_loss_both = torch.nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor([1,2.75]))
custom_class_loss_focal = FocalLoss(torch.nn.CrossEntropyLoss()) 
def fastercnn_loss_ls(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = custom_class_loss_ls(class_logits, labels)# F.cross_entropy(class_logits, labels, label_smoothing=label_smoothing, weight=pos_weights)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    # print('custom Loss')
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def fastercnn_loss_focal(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = custom_class_loss_focal(class_logits, labels)#sigmoid_focal_loss(class_logits, labels)# F.cross_entropy(class_logits, labels, label_smoothing=label_smoothing, weight=pos_weights)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    # print('custom Loss')
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def fastercnn_loss_cw(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = custom_class_loss_cw(class_logits, labels)# F.cross_entropy(class_logits, labels, label_smoothing=label_smoothing, weight=pos_weights)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    # print('custom Loss')
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def fastercnn_loss_both(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = custom_class_loss_both(class_logits, labels)# F.cross_entropy(class_logits, labels, label_smoothing=label_smoothing, weight=pos_weights)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    # print('custom Loss')
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

