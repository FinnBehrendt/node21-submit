import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
import numpy as np 
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics import ROC, Accuracy, MetricCollection, Precision, Recall, AUROC, AveragePrecision, F1, AUC, PrecisionRecallCurve, Specificity, ConfusionMatrix
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
from src.models.modules.DETR.detr import PostProcess
def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = False):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder path
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):

                # don't upload files ignored by git
                # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
                command = ["git", "check-ignore", "-q", str(path)]
                not_ignored = subprocess.run(command).returncode == 1

                # don't upload files from .git folder
                not_git = not str(path).startswith(str(git_dir_path))

                if path.is_file() and not_git and not_ignored:
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, prefix ):
        self.preds = []
        self.targets = []
        self.ready = True
        # self.num_classes = num_classes
        self.prefix = prefix
        # self.pathologies = class_names

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment
            # Rearranging dicts
            max_preds = []
            target_scores = []
            boxes_target = []
            boxes_pred = []

            [boxes_target.extend(target) for target in self.targets]
            [boxes_pred.extend(pred) for pred in self.preds]

            for preds,targets in zip(boxes_pred,boxes_target) : # preds and targets
                max_preds.append(preds['scores'].max())
                target_scores.append(targets['labels'].max())

            metrics = ConfusionMatrix(num_classes=2).to(torch.device(trainer.model.device))
            confusion_matrix = metrics(torch.stack(max_preds),torch.stack(target_scores)).cpu().numpy()
            
            # set figure size
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        
            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(confusion_matrix,ax=ax, annot=True, annot_kws={"size": 15}, fmt="g",cbar=False)
                # ax[mat].set_title(self.pathologies[mat])
            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"{self.prefix}confusion_matrix/val": wandb.Image(fig)}, commit=False)


            # reset plot
            plt.clf()
            plt.cla()
            
            self.preds.clear()
            self.targets.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self,num_classes=2, prefix = None):
        self.preds = []
        self.targets = []
        self.ready = True
        # self.class_names = 2
        self.prefix = prefix
        self.num_classes = num_classes
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # Rearranging dicts
            max_preds = []
            target_scores = []
            boxes_target = []
            boxes_pred = []

            [boxes_target.extend(target) for target in self.targets]
            [boxes_pred.extend(pred) for pred in self.preds]

            for preds,targets in zip(boxes_pred,boxes_target) : # preds and targets
                max_preds.append(preds['scores'].max())
                target_scores.append(targets['labels'].max())            
            metric_collection = MetricCollection([
                            AUROC(),
                            AveragePrecision(),
                            Accuracy(threshold=0.5),
                            Precision(threshold=0.5),
                            Recall(threshold=0.5),
                            F1(threshold=0.5),
                            Specificity(threshold=0.5)
                                ]).to(torch.device(trainer.model.device))

            metrics = metric_collection(torch.stack(max_preds),torch.stack(target_scores))

            data = [metrics['F1'].cpu().numpy(),metrics['Precision'].cpu().numpy() , metrics['Recall'].cpu().numpy(), metrics['AUROC'].cpu().numpy(), metrics['AveragePrecision'].cpu().numpy(), metrics['Accuracy'].cpu().numpy(), metrics['Specificity'].cpu().numpy()]
            data = [dat.item() for dat in data]
            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                np.array(data).reshape(7,1),
                annot=True,
                annot_kws={"size": 13},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall","AUROC", "AveragePrecision", "Accuracy","Specificity"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"{self.prefix}f1_p_r_heatmap/val": wandb.Image(plt)}, commit=False)

            # reset plot
            plt.clf()
            plt.cla()
            plt.close('all')
            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, prefix = None):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.prefix = prefix
        self.prediction_confidence_threshold = 0.05

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            # val_imgs, val_labels = val_samples['img'], val_samples['lab']
            is_effdet = False
            is_detr = False
            if len(val_samples) == 3:
                image, annotations, targets = val_samples
                is_effdet = True
                image = torch.cat([image,image,image],1) # to rgb
                image = image.to(device=pl_module.device)
      
                annotations['bbox'] = [x.to(device=pl_module.device) for x in annotations['bbox']]
                annotations['cls'] = [x.to(device=pl_module.device) for x in annotations['cls']]
                annotations['img_size'] = annotations['img_size'].to(device=pl_module.device)  
                annotations['img_scale'] = annotations['img_scale'].to(device=pl_module.device)  
            elif pl_module.cfg.name == 'DETR':
                is_detr = True
                image, targets = val_samples
                targets = [{k: v for k, v in t.items()} for t in targets]
                image = list(im.to(device=pl_module.device) for im in image)
                for i, target in enumerate(targets):
                    if len(target['labels']) == 1:
                        if target['labels'] == 0:
                            targets[i]['labels'] = torch.tensor([],device=target['labels'].device)

                image = [torch.stack([x,x,x],1) for x in image] # to rgb
                image = torch.stack(image,0).squeeze()

            else: 
                image, targets = val_samples
                # run the batch through the network
                image = [im.to(device=pl_module.device) for im in image]
            if is_effdet:
                logits_dict = pl_module(image,annotations)
                detections = logits_dict['detections'] 
       
                # convert tensors to list as in FasterCNN!
                # list on len=batch size
                logits_dict = []
                for i in range(detections.shape[0]):
                    postprocessed = self._postprocess_single_prediction_detections(detections[i]) #return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
                    boxes = postprocessed['boxes']
                    scores = postprocessed['scores']
                    if len(scores) == 0: 
                        scores = torch.tensor([0.0],device=trainer.model.device)
                    # boxes = detections[i,:,0:4]
                    # scores = detections[i,:,4]
                    logits_dict.append({'boxes':boxes , 'scores':scores })
            elif pl_module.cfg.name == 'DETR':
                output = pl_module(image)
                orig_target_sizes = torch.stack([torch.tensor(t['img_size'],device=pl_module.device) for t in targets], dim=0)
                postprocess = PostProcess()
                logits_dict = postprocess(output, orig_target_sizes)
            else: 
                logits_dict = pl_module(image)

            logits = logits_dict[:self.num_samples]
            targets = targets[:self.num_samples]
            for i in range(self.num_samples):
                if len(targets[i]['boxes'])>0:
                    image[i] = draw_bounding_boxes((image[i]*255).type(torch.uint8).cpu(),targets[i]['boxes'].cpu(),colors='red',width= 2).type(torch.float32)
                if len(logits[i]['boxes'])>0:
                    image[i] = draw_bounding_boxes((image[i]*255).type(torch.uint8).cpu(),logits[i]['boxes'].cpu(),colors='green',width= 2, labels=[str(x.item()) for x in logits[i]['scores'].cpu()], font_size=15).type(torch.float32)
                # image_box.append(image[i].type(torch.float32))
            # log the images as wandb Image
            experiment.log(
                {
                    f"{self.prefix}Images/val": [
                        wandb.Image(x, caption=f"Scores: {[x.__format__('.2f') for x in pred['scores']]}, Labels: {y['labels'].cpu().numpy()}")
                        for x, pred, y in zip(
                            image[: self.num_samples],
                            logits[: self.num_samples],
                            targets[: self.num_samples],
                        )
                    ]
                }
            )
            plt.cla()
            plt.clf()
    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        indexes = torch.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}