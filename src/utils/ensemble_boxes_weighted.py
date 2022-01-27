import pandas as pd
import torch 
import pickle
import numpy as np
from ensemble_boxes import *

def ensemble_boxes(outs,fusing='wbf', weights = None, iou_thresh = 0.2, skip_boxes_thresh = 0.001,sigma = 0.1):
    # with open('/home/Behrendt/test.pkl','rb') as f:
    #     outs = pickle.load(f)
    preds = outs['test-preds']
    targets = outs['test-targets']
    out_dict = {'test-preds':[],'test-targets':targets[0]}
    boxes_pred = []
    boxes_target = []
    labels = []
    scores = []

    for i in range(len(preds[0])) : # TODO check this
        for j in range(len(preds)):
            boxes_pred.append(preds[j][i]['boxes']) # 1 bild 
            boxes_target.append(targets[j][i]['boxes'] )
            
            scores.append(preds[j][i]['scores'])
            labels.append(targets[j][i]['labels'])

    boxes_pred = np.reshape(boxes_pred,[-1,len(preds)]).T
    boxes_target = np.reshape(boxes_target,[-1,len(preds)]).T
    scores = np.reshape(scores,[-1,len(preds)]).T
    labels = np.reshape(labels,[-1,len(preds)]).T
    new_boxes = []
    new_scores = []
    for k in range(len(preds[0])) :
        # if len(boxes_pred)>0:
        # if list(boxes_pred[:,k])[0]
        boxes_list = [list(x.tolist()) for x in list(boxes_pred[:,k])]
        scores_list = [list(x.tolist()) for x in list(scores[:,k])]
        labels_list = [list(x.tolist()) for x in list(labels[:,k])]
        # else: 
        #     boxes_list = list(boxes_pred[:,k])
        #     scores_list = list(scores[:,k])
        #     labels_list = list(labels[:,k])
        # boxes
        # if np.random.random()>0.5:
        #     boxes_list = [[] for x in boxes_list]
        if np.sum([len(x) for x in boxes_list])>0:
            
            # the labels dont really matter
            for i, label in enumerate(labels_list):
                if len(label)< len(boxes_list[i]):
                    labels_list[i].extend([k] * (len(boxes_list[i])-len(label)))
                elif len(label)>len(boxes_list[i])    :
                    labels_list[i] = labels_list[i][:len(boxes_list[i])]

            pop_list = []
            for j in range(len(scores_list)):
                if len(scores_list[j])>0:
                    pop_list.append(j)
            boxes_list = [boxes_list[x] for x in pop_list]
            scores_list = [scores_list[x] for x in pop_list]
            labels_list = [labels_list[x] for x in pop_list]
            if weights is not None:
                weight = weights[k,pop_list]
            else: 
                weight = weights
            for i in range(len(boxes_list)): 
                for j in range(len(boxes_list[i])):
                    boxes_list[i][j] = [x/1024 for x in  boxes_list[i][j]]
        else: 
            boxes_list = [[] for x in boxes_list]
            scores_list =[[] for x in scores_list]
            labels_list =[[] for x in labels_list]
            weight = None
        if fusing == 'wbf':
            boxes, weighted_scores, weighted_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh, skip_box_thr=skip_boxes_thresh)
        elif fusing == 'nms':
            boxes, weighted_scores, weighted_labels = nms(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh)
        elif fusing == 'soft_nms':
            boxes, weighted_scores, weighted_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh, sigma=sigma)
        new_boxes.append(boxes*1024)
        new_scores.append(weighted_scores)

    for i,pred in enumerate(preds[0]):
        pred['boxes'] = torch.tensor(new_boxes[i])
        pred['scores'] = torch.tensor(new_scores[i])
        pred['labels'] = torch.tensor(0)
        out_dict['test-preds'].append(pred)
    # print('rone')
    return out_dict