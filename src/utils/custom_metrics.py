from torchvision.ops import box_iou
import torch
from torchmetrics import ROC
import matplotlib.pylab as plt
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import src.utils.utils as utils

# Challenge Metrics

def calc_iou(pred, target):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

def calc_FROC(preds, targets):
    # We assume, that NMS is already performed!
    num_images = len(preds)
    num_lesions = 0
    num_candidates = 0 
    all_fps = []
    all_tps = []
    for pred, target  in zip(preds,targets):
        #pred = utils.get_NonMaxSup_boxes(pred)
        num_candidates += 1
        num_lesions  += target['boxes'].shape[0]
        tp, fp = compute_fp_tp_per_image(pred,target)
        # print('num_lesions',target['boxes'].shape[0])
        # print('tp',tp)
        # print('fp',fp)
        if target['boxes'].shape[0] < len(tp):
            print('more tp than targets: tps:', tp, 'targets ', target['boxes'], 'index', num_candidates)
        all_tps.extend(tp)
        all_fps.extend(fp)

    if num_lesions < len(tp):
        print('ERROR')
    fp_per_img, sens_total = compute_froc_curve_data(np.array([x.cpu().numpy() for x in all_fps]),np.array([x.cpu().numpy() for x in all_tps]),num_lesions,num_images)
    return fp_per_img, sens_total




def compute_fp_tp_per_image(preds,targets):
    scores = preds['scores']
    labels = targets['labels']
    boxes_pred = preds['boxes']
    boxes_true = targets['boxes']
    num_targets = len(labels)
    num_preds = len(scores)
    thresh = 0.2

    mat = torch.zeros([len(scores),len(labels)])

    if len(boxes_pred) == 0 and len(boxes_true) == 0:
        tp, fp = [], []
    elif len(boxes_pred) == 0 and len(boxes_true) != 0:
        tp, fp = [], []
    elif len(boxes_pred) != 0 and len(boxes_true) == 0:
        tp, fp = [], scores
    else: 
        tp = []
        fp = []
        for i, score in enumerate(scores): 
            for j, label in enumerate(labels):
                    iou = box_iou(boxes_pred[i].unsqueeze(0),boxes_true[j].unsqueeze(0)).item()
                    if iou > thresh :
                        mat[i,j] = torch.tensor(scores[i])
        for j in range(mat.shape[1]):
            if len(mat[:,j][mat[:,j]>0])>1: # wenn mehr als ein Eintrag>0 behalte den größten und setze die anderen auf 0
                max_score = mat[:,j][mat[:,j]>0].max()
                max_ind = mat[:,j].argmax()
                mat[:,j] = torch.zeros_like(mat[:,j])
                mat[:,j][max_ind] =  max_score
                
        for i in range(mat.shape[0]):
            if mat[i,:].sum().item() == 0 :
                fp.append(scores[i])
            elif mat[i,:].sum().item() > 0: 
                tp.append(scores[i])
    return tp, fp 



def compute_fp_tp_probs(
    probs: Union[np.ndarray, torch.Tensor],
    y_coord: Union[np.ndarray, torch.Tensor],
    x_coord: Union[np.ndarray, torch.Tensor],
    evaluation_mask: Union[np.ndarray, torch.Tensor],
    labels_to_exclude: Optional[List] = None,
    resolution_level: int = 0,
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to distinguish
    true positive and false positive predictions. A true positive prediction is defined when
    the detection point is within the annotated ground truth region.
    Args:
        probs: an array with shape (n,) that represents the probabilities of the detections.
            Where, n is the number of predicted detections.
        y_coord: an array with shape (n,) that represents the Y-coordinates of the detections.
        x_coord: an array with shape (n,) that represents the X-coordinates of the detections.
        evaluation_mask: the ground truth mask for evaluation.
        labels_to_exclude: labels in this list will not be counted for metric calculation.
        resolution_level: the level at which the evaluation mask is made.
    Returns:
        fp_probs: an array that contains the probabilities of the false positive detections.
        tp_probs: an array that contains the probabilities of the True positive detections.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.
    """
    if not (probs.shape == y_coord.shape == x_coord.shape):
        raise AssertionError("the shapes for coordinates and probabilities should be the same.")

    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(y_coord, torch.Tensor):
        y_coord = y_coord.detach().cpu().numpy()
    if isinstance(x_coord, torch.Tensor):
        x_coord = x_coord.detach().cpu().numpy()
    if isinstance(evaluation_mask, torch.Tensor):
        evaluation_mask = evaluation_mask.detach().cpu().numpy()

    if labels_to_exclude is None:
        labels_to_exclude = []

    max_label = np.max(evaluation_mask)
    tp_probs = np.zeros((max_label,), dtype=np.float32)

    y_coord = (y_coord / pow(2, resolution_level)).astype(int)
    x_coord = (x_coord / pow(2, resolution_level)).astype(int)

    hittedlabel = evaluation_mask[y_coord, x_coord]
    fp_probs = probs[np.where(hittedlabel == 0)]
    for i in range(1, max_label + 1):
        if i not in labels_to_exclude and i in hittedlabel:
            tp_probs[i - 1] = probs[np.where(hittedlabel == i)].max()

    num_targets = max_label - len(labels_to_exclude)
    return fp_probs, tp_probs, num_targets


def compute_froc_curve_data(
    fp_probs: Union[np.ndarray, torch.Tensor],
    tp_probs: Union[np.ndarray, torch.Tensor],
    num_targets: int,
    num_images: int,
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the required data for plotting the Free Response Operating Characteristic (FROC) curve.
    Args:
        fp_probs: an array that contains the probabilities of the false positive detections for all
            images under evaluation.
        tp_probs: an array that contains the probabilities of the True positive detections for all
            images under evaluation.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.
        num_images: the number of images under evaluation.
    """
    if not isinstance(fp_probs, type(tp_probs)):
        raise AssertionError("fp and tp probs should have same type.")
    if isinstance(fp_probs, torch.Tensor):
        fp_probs = fp_probs.detach().cpu().numpy()
    if isinstance(tp_probs, torch.Tensor):
        tp_probs = tp_probs.detach().cpu().numpy()

    total_fps, total_tps = [], []
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)))
    for thresh in all_probs[1:]:
        total_fps.append((fp_probs >= thresh).sum())
        total_tps.append((tp_probs >= thresh).sum())
    total_fps.append(0)
    total_tps.append(0)
    fps_per_image = np.asarray(total_fps) / float(num_images)
    total_sensitivity = np.asarray(total_tps) / float(num_targets)
    return fps_per_image, total_sensitivity


def compute_froc_score(
    fps_per_image: np.ndarray, total_sensitivity: np.ndarray, eval_thresholds: Tuple = (0.25, 0.5, 1, 2, 4, 8)
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the challenge's second evaluation metric, which is defined as the average sensitivity at
    the predefined false positive rates per whole slide image.
    Args:
        fps_per_image: the average number of false positives per image for different thresholds.
        total_sensitivity: sensitivities (true positive rates) for different thresholds.
        eval_thresholds: the false positive rates for calculating the average sensitivity. Defaults
            to (0.25, 0.5, 1, 2, 4, 8) which is the same as the CAMELYON 16 Challenge.
    """
    interp_sens = np.interp(eval_thresholds, fps_per_image[::-1], total_sensitivity[::-1])
    return np.mean(interp_sens)