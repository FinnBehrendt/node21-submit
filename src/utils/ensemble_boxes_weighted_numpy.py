from unicodedata import name
import pandas as pd
import torch 
import pickle
import numpy as np
from ensemble_boxes import *
from src.utils.custom_metrics import calc_iou, calc_FROC
from monai.metrics import compute_froc_score
import wandb 
import time 
import matplotlib.pyplot as plt 
from torchmetrics import MetricCollection, AUROC, MAP, ROC, PrecisionRecallCurve
def ensemble_boxes(outs,fusing='wbf', weights = None, iou_thresh = 0.2, skip_boxes_thresh = 0.001,sigma = 0.1):
    # with open('/home/Behrendt/test.pkl','rb') as f:
    #     outs = pickle.load(f)
    preds = outs
    
    # targets = outs['test-targets']
    # out_dict = {'test-preds':[],'test-targets':targets[0]}
    boxes_pred = []
    boxes_target = []
    labels = []
    scores = []
    slices_pred = []
    for i in range(len(preds[0])) :
        for j in range(len(preds)):
            boxes_pred.append(preds[j][i]['boxes']) # 1 bild 
            slices_pred.append(preds[j][i]['slice']) # 1 bild 
            
            scores.append(preds[j][i]['scores'])
            labels.append(np.array([1]*len(preds[j][i]['scores'])))
    # print(len(preds),len(preds[0]))
    # print(len(boxes_pred[0]))
    # print(len(boxes_pred[0][0]))
    # print(boxes_pred[0][0])
    # print(boxes_pred[0][0].dtype)
    # boxes_pred[0]=boxes_pred[0][:-1]
    # scores[0]=scores[0][:-1]
    # labels[0]=labels[0][:-1]
    # slices_pred[0]=slices_pred[0][:-1]
    boxes_pred = np.reshape(boxes_pred,[-1,len(preds)],).T
    scores = np.reshape(scores,[-1,len(preds)]).T
    labels = np.reshape(labels,[-1,len(preds)]).T
    slices_pred = np.reshape(slices_pred,[-1,len(preds)]).T
    new_boxes = []
    new_scores = []
    new_slices = [] 
    for k in range(len(preds[0])) :
        # if len(boxes_pred)>0:
        # if list(boxes_pred[:,k])[0]
        # print(boxes_pred.shape)
        boxes_list = [list(x) for x in list(boxes_pred[:,k])]
        scores_list = [list(x) for x in list(scores[:,k])]
        labels_list = [list(x) for x in list(labels[:,k])]
        slices_list = [list(x) for x in list(slices_pred[:,k])]
        # else: 
        #     boxes_list = list(boxes_pred[:,k])
        #     scores_list = list(scores[:,k])
        #     labels_list = list(labels[:,k])
        # boxes
        # if np.random.random()>0.5:
        #     boxes_list = [[] for x in boxes_list]
        if np.sum([len(x) for x in boxes_list])>0:
            
            # the labels dont really matter
            # for i, label in enumerate(labels_list):
            #     if len(label)< len(boxes_list[i]):
            #         labels_list[i].extend([k] * (len(boxes_list[i])-len(label)))
            #     elif len(label)>len(boxes_list[i])    :
            #         labels_list[i] = labels_list[i][:len(boxes_list[i])]

            pop_list = []
            for j in range(len(scores_list)):
                if len(scores_list[j])>0:
                    pop_list.append(j)
            boxes_list = [boxes_list[x] for x in pop_list]
            scores_list = [scores_list[x] for x in pop_list]
            labels_list = [labels_list[x] for x in pop_list]
            slices_list = [slices_list[x] for x in pop_list]
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
            slices_list =[[] for x in slices_list]
            weight = None
        if fusing == 'wbf':
            boxes, weighted_scores, weighted_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh, skip_box_thr=skip_boxes_thresh)
        elif fusing == 'nms':
            boxes, weighted_scores, weighted_labels = nms(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh)
        elif fusing == 'soft_nms':
            boxes, weighted_scores, weighted_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weight, iou_thr=iou_thresh, sigma=sigma)
        elif fusing == 'nmw':
            boxes, weighted_scores, weighted_labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weight, skip_box_thr=skip_boxes_thresh, iou_thr=iou_thresh)
        new_boxes.append(boxes*1024)
        new_scores.append(weighted_scores)
        new_slices.append([k]*len(boxes))
    predictions = []
    for i,pred in enumerate(preds[0]):
        pred['boxes'] = new_boxes[i]
        pred['scores'] = new_scores[i]
        pred['slice'] = new_slices[i]
        predictions.append(pred)
    # print('rone')
    return predictions

def eval_ensemble(boxes_pred,boxes_target, name='TestRun'):
    # boxes_pred, boxes_target = ouputs_ensemble['test-preds'], ouputs_ensemble['test-targets']
    max_preds = []
    target_scores = []

    for preds,targets in zip(boxes_pred,boxes_target) : # preds and targets
        if len(preds['scores']) == 0: 
            max_preds.append(0.0)
        else:
            max_preds.append(preds['scores'].max())
        target_scores.append(targets['labels'].max().item())

    metric_collection_classification = MetricCollection(AUROC())
    metric_class = metric_collection_classification(torch.tensor(max_preds),torch.tensor(target_scores))    
    keys = ['scores', 'boxes']
    for key in keys:
        for i, box in enumerate(boxes_pred):
            boxes_pred[i][key] = torch.tensor(boxes_pred[i][key])
    # boxes_pred = {str(key):[torch.tensor(i) for i in val]
    #         for key, val in boxes_pred.items()}
    fps, sens = calc_FROC(boxes_pred, boxes_target)
    froc_0125 = compute_froc_score(fps,sens,eval_thresholds = (0.125))
    froc_025 = compute_froc_score(fps,sens,eval_thresholds = (0.25))
    froc_05 = compute_froc_score(fps,sens,eval_thresholds = (0.5))
    froc_all = compute_froc_score(fps,sens,eval_thresholds = (0.125,0.25,0.5))
    roc = ROC()
    prc = PrecisionRecallCurve()
    fpr, tpr, thresholds_roc = roc(torch.tensor(max_preds),torch.tensor(target_scores))
    prec, rec, thresholds_prc = prc(torch.tensor(max_preds),torch.tensor(target_scores))

    comp_metric = 0.75 * metric_class['AUROC'] + 0.25 * froc_025
    out_dict = {'FROC_all': froc_all, 'FROC_0125': froc_0125, 'FROC_025': froc_025, 'FROC_05': froc_05, 'AUROC':metric_class['AUROC'], 'Competition_metric': comp_metric }
    wandb.init(project='ensembling-node21',name=name)
    time.sleep(5)
    for met in out_dict:
        # log metrics
        wandb.log({f'fold-ensemble-ckpt/{met}/':out_dict[met]})
        time.sleep(3)
    # Plot FROC Curve
    data_roc = [[x, y] for (x, y) in zip(fps,sens)]
    table_roc = wandb.Table(data=data_roc, columns = ["fps", "sens"])
    wandb.log({f'fold-ensemble-ckpt/PLOTS/FROC/' : wandb.plot.line(table_roc, "fps", "sens",title=f'fold-ensemble-ckpt/PLOTS/FROC/')})
    plt.clf()
    plt.cla()
    plt.close('all')
    time.sleep(5)    
        # Plot and Log ROC and PRC Curve to wandb

    data_roc = [[x, y] for (x, y) in zip(tpr.cpu(),fpr.cpu())]
    table_roc = wandb.Table(data=data_roc, columns = ["tpr", "fpr"])
    wandb.log({f'fold-ensemble-ckpt/PLOTS/AUC/' : wandb.plot.line(table_roc, "fpr", "tpr",title=f'fold-ensemble-ckpt/PLOTS/AUC/')})
    plt.clf()
    plt.cla()
    plt.close('all')
    time.sleep(5)
    data_prc = [[x, y] for (x, y) in zip(prec.cpu(),rec.cpu())]
    table_prc = wandb.Table(data=data_prc, columns = ["Precision", "Recall"])
    wandb.log({f'fold-ensemble-ckpt/PLOTS/PRC/' : wandb.plot.line(table_prc, "Precision", "Recall",title=f'fold-ensemble-ckpt/PLOTS/PRC/')})
    plt.clf()
    plt.cla()
    plt.close('all')


    return out_dict, sens, fps, fpr, tpr, prec, rec