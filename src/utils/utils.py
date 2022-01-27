import logging
import os
import warnings
from typing import List, Sequence
import torch
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import yaml
import numpy as np 
def get_yaml(path): # read yaml 
    with open(path, "r") as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return file


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "experiment"
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams['run_id'] = trainer.logger.experiment[0].id
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def get_checkpoint(cfg, path): 
    checkpoint_path = path
    checkpoint_to_load = cfg.get("checkpoint",'last') # default to last.ckpt 
    all_checkpoints = os.listdir(checkpoint_path + '/checkpoints')
    # wandbID = all_checkpoints[0].split('wandbID-')[1].split('.ckpt')[0] # Get wandb ID
    hparams = get_yaml(path+'/csv/fold-1/hparams.yaml')
    wandbID = hparams['run_id']
    checkpoints = {'fold-1':[],
                    'fold-2':[],
                    'fold-3':[],
                    'fold-4':[],
                    'fold-5':[]} # dict to store the checkpoints with their path for different folds

    if checkpoint_to_load == 'last':
        matching_checkpoints = [c for c in all_checkpoints if "last" in c]
        for fold, cp_name in enumerate(matching_checkpoints):
            checkpoints[f'fold-{fold+1}'] = checkpoint_path + '/checkpoints/' + cp_name
    elif 'best' in checkpoint_to_load :
        matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
        matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4]) # sort by loss value -> increasing
        for fold in checkpoints:
            for cp in matching_checkpoints:
                if f"{fold}" in cp:
                    checkpoints[fold].append(checkpoint_path + '/checkpoints/' + cp)
            if not 'best_k' in checkpoint_to_load: # best_k loads the k best checkpoints 
                checkpoints[fold] = checkpoints[fold][0] # get only the best (first) checkpoint of that fold
    return wandbID, checkpoints

def collate_fn(batch):
    return tuple(zip(*batch))


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (float(boxAArea + boxBArea - interArea) + 0.0000000001)
    # return the intersection over union value
    return iou

def get_NonMaxSup_boxes(pred_dict):
    scores = pred_dict['scores']
    boxes = pred_dict['boxes']
    lambda_nms = 0.2

    out_scores = []
    out_boxes = []
    for ix, (score, box) in enumerate(zip(scores,boxes)):
        discard = False
        for other_box in out_boxes:
            if intersection_over_union(box, other_box) > lambda_nms:
                discard = True
                break
        if not discard:
            out_scores.append(score)
            out_boxes.append(box)
    if out_boxes:
        try:
            out_scores = torch.stack(out_scores)
            out_boxes = torch.stack(out_boxes)
        except: 
            out_scores = np.stack(out_scores)
            out_boxes = np.stack(out_boxes)            
    else: 
        try: 
            out_scores = torch.tensor([])
            out_boxes =  torch.tensor([])
        except: 
            out_scores = np.array([])
            out_boxes =  np.array([])
    return {'scores':out_scores, 'boxes':out_boxes}

def xyxy2yxyx(detections): # converts xyxy in yxyx format
    for i, det in enumerate(detections):
        for j, box in enumerate(det['boxes']): 
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            detections[i]['boxes'][j]  = torch.tensor([ymin, xmin, ymax, xmax ],device=box.device) 
    return detections

def yxyx2xyxy(detections): # converts yxyx in xyxy format -- actaully does the same as xyxy2yxyx
    for i, det in enumerate(detections):
        for j, box in enumerate(det['boxes']): 
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            detections[i]['boxes'][j] = torch.tensor([xmin, ymin, xmax ,ymax] ,device=box.device) 
    return detections