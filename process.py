from genericpath import exists
import SimpleITK
import numpy as np

from pandas import DataFrame
from scipy.ndimage import center_of_mass, label
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from wandb import Config
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from collections import OrderedDict
from skimage import transform
from torchvision import transforms
import json
from typing import Dict
import math
import albumentations as A
from typing import List, Optional
# import training_utils.utils as utils
# from training_utils.dataset import CXRNoduleDataset, get_transform
import os
# from training_utils.train import train_one_epoch
import itertools
from pathlib import Path
from postprocessing import get_NonMaxSup_boxes, PostProcess
from omegaconf import DictConfig
from src.utils.utils import xyxy2yxyx, yxyx2xyxy
# Config stuff
import hydra
from omegaconf import DictConfig, OmegaConf
import glob
# import lightning modules
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.plugins import DDPPlugin
import pandas as pd
from src.utils.ensemble_boxes_weighted_numpy import ensemble_boxes
from sklearn.model_selection import StratifiedGroupKFold
from training_utils.yolov5.train import main as train_yolo 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
import sys
import yaml 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
# update
# update
# update

execute_in_docker = False 

if not execute_in_docker :
    print('NOT IN DOCKER MODE') 

class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=False, retrain=False, retest=False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(input_dir) if execute_in_docker else Path('//home/Behrendt/projects/Node21/node21-submit/test/'),
            output_file = Path(os.path.join(output_dir,'nodules.json')) if execute_in_docker else Path(os.path.join('//home/Behrendt/projects/Node21/node21-submit/output/','nodules.json'))

        )
        self.retrain = retrain
        ### Model ###


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('using the device ', self.device)
        print(torch.__version__)
        self.input_path, self.output_path = input_dir, output_dir

        self.model_ckpt_dir = "/opt/algorithm/checkpoints/retrain/" if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/output/retrain/"
        path_prefix_base = Path("/opt/algorithm/checkpoints/") if execute_in_docker else Path("//home/Behrendt/projects/Node21/node21-submit/checkpoints/")
        if not retest:
            path_prefix = {
                            'fcrnn_l': os.path.join(path_prefix_base,Path("notest_final/fcrnn_1024_gen//")), # all possible architectures
                            'retina_l':  os.path.join(path_prefix_base,Path("notest_final/retina_1024_gen//")),
                            'effdet2_l':  os.path.join(path_prefix_base,Path("notest_final/effdet_1024_gen/")),
                            'yolo_s':  os.path.join(path_prefix_base,Path("notest_final/yolo_640_gen")),
                            # 'fcrnn_s': os.path.join(path_prefix_base,Path("notest_final/fcrnn_512_gen//")), # all possible architectures
                            # 'retina_s':  os.path.join(path_prefix_base,Path("notest_final/retina_512_gen//")),
                            # 'effdet2_s':  os.path.join(path_prefix_base,Path("notest_final/effdet_512_gen/")),
                            'yolo_l':  os.path.join(path_prefix_base,Path("notest_final/yolo_1024_gen")),
                            # 'detr':  os.path.join(path_prefix_base,Path("final_models/detr_1024_gen"))
                            }
        else: # TODO
             path_prefix = {
                            'fcrnn_l': os.path.join(self.model_ckpt_dir,Path("fcrnn_l/")), # all possible architectures
                            'retina_l':  os.path.join(self.model_ckpt_dir,Path("retina_l")),
                            'effdet2_l':  os.path.join(self.model_ckpt_dir,Path("effdet2_l")),
                            'yolo_s':  os.path.join(self.model_ckpt_dir,Path("yolo_s")),
                            # 'fcrnn_s': os.path.join(self.model_ckpt_dir,Path("fcrnn_s")), # all possible architectures
                            # 'retina_s':  os.path.join(self.model_ckpt_dir,Path("retina_s")),
                            # 'effdet2_s':  os.path.join(self.model_ckpt_dir,Path("effdet2_s")),
                            'yolo_l':  os.path.join(self.model_ckpt_dir,Path("yolo_l")),
                            # 'detr':  os.path.join(self.model_ckpt_dir,Path("detr"))
        }
        checkpoints = {'fold-1':[],
                'fold-2':[],
                'fold-3':[],
                'fold-4':[],
                'fold-5':[]
                } # dict to store the checkpoints with their path for different folds

        archs = {
            'fcrnn_l': checkpoints, # all possible architectures
            'retina_l': checkpoints,
            'effdet2_l': checkpoints,
            'yolo_s': checkpoints,
            # 'fcrnn_s': checkpoints, # all possible architectures
            # 'retina_s': checkpoints,
            # 'effdet2_s': checkpoints,
            'yolo_l': checkpoints,
            # # 'detr': checkpoints
                }

        self.models = {}
        self.ensemble_per_model = False
        for arch in archs: 
            if 'yolo' in str(path_prefix[arch]).lower():
                checkpoint_to_load = 'best_k'
            else: 
                checkpoint_to_load = 'last'
            print(f'LOADING {checkpoint_to_load} Checkpoints for {arch}')
            archs[arch] = {'fold-1':[],
                    'fold-2':[],
                    'fold-3':[],
                    'fold-4':[],
                    'fold-5':[]
                    }
            self.models[arch] = {'fold-1':[],
                    'fold-2':[],
                    'fold-3':[],
                    'fold-4':[],
                    'fold-5':[]
                    }

            all_checkpoints = os.listdir(path_prefix[arch])
            best_ckpts = [] 


            if checkpoint_to_load == 'last':
                matching_checkpoints = [c for c in all_checkpoints if "last" in c]
                for fold, cp_name in enumerate(matching_checkpoints):
                    archs[arch][f'fold-{fold+1}'] = [Path(os.path.join(path_prefix[arch],Path(cp_name)))]
            elif 'best_k' in checkpoint_to_load :
                matching_checkpoints = [c for c in all_checkpoints if "last" not in c]
                # matching_checkpoints.sort(key = lambda x: x.split('loss-')[1][0:4]) # sort by loss value -> increasing
                for fold in checkpoints:
                    for cp in matching_checkpoints:
                        if f"{fold}" in cp:
                            archs[arch][fold].append(Path(os.path.join(path_prefix[arch],Path(cp))))

            elif checkpoint_to_load == 'best':
                for fold in archs[arch]:
                    for cp in best_ckpts:
                        if f"{fold}" in cp:
                            archs[arch][fold].append(cp)

            
            if 'effdet' in str(path_prefix[arch]).lower():
                del archs[arch]['fold-1']
                del self.models[arch]['fold-1'] 
                del archs[arch]['fold-5']
                del self.models[arch]['fold-5'] 
                del archs[arch]['fold-4']
                del self.models[arch]['fold-4'] 
                del archs[arch]['fold-3']
                del self.models[arch]['fold-3'] 
            elif 'yolo_l' in str(path_prefix[arch]).lower():
                del archs[arch]['fold-2']
                del self.models[arch]['fold-2']
                
            for fold in archs[arch]: # for all models
                self.effdet = False 
                self.yolo = False
                self.detr = False

                for ckpt in archs[arch][fold]: # for all individual checkpoints

                        if 'effdet' in str(ckpt).lower(): # important to name folder correct..
                            from src.models.Detector_effdet import Detector as Detector 
                            self.effdet = True
                            self.model = Detector.load_from_checkpoint(ckpt,map_location=self.device)
                            print('USING E')
                        elif 'detr' in str(ckpt).lower():
                            from src.models.Detector_detr import Detector as Detector 
                            self.detr = True
                            self.postprocessors = PostProcess()
                            self.model = Detector.load_from_checkpoint(ckpt,map_location=self.device)
                            print('USING D')
                        elif 'yolo' in str(ckpt).lower():       
                            modelpath = Path("/opt/algorithm/training_utils/yolov5") if execute_in_docker else Path("//home/Behrendt/projects/Node21/node21-submit/training_utils/yolov5/")
                            self.model = torch.hub.load(modelpath, 'custom', path=ckpt, autoshape=True,force_reload=True, source='local') # local model
                            self.yolo = True
                            print('USING Y')
                            self.model.conf = 0.01 # 0.25
                            self.model.iou = 0.2 # 0.45
                        else: 
                            from src.models.Detector import Detector as Detector
                            self.effdet = False
                            self.model = Detector.load_from_checkpoint(ckpt,map_location=self.device)

                    
                        self.model.to(self.device)
                        self.models[arch][fold].append(self.model)
        if retrain: 
            self.archs = archs
        
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)
            
    # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results. 
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225) 
        is overwritten here, so that it directly returns the predictions without changing the format.
        
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)
        
        # Write resulting candidates to nodules.json for this case
        return scored_candidates
    
   
    
    #--------------------Write your retrain function here ------------
    def train(self, num_epochs = 1):
        '''
        input_dir: Input directory containing all the images to train with
        output_dir: output_dir to write model to.
        num_epochs: Number of epochs for training the algorithm.
        '''
        # create training dataset and defined transformations
        for modelname in self.models:
            yolo = 'yolo' in modelname
            ## Load Data 
            input_dir = self.input_path if execute_in_docker else Path("//home/Behrendt/projects/Node21/node21-submit/input_train/")
            output_dir = self.output_path if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/output/"
            # save models to 
            model_ckpt_dir = self.model_ckpt_dir + modelname
            # load meatadata from
            path_data = os.path.join(input_dir, 'metadata.csv')
            # make model paths if they dont exist
            os.makedirs(model_ckpt_dir,exist_ok=True)

            # load metadata and create train / val / test set(s)
            train_df = pd.read_csv(path_data)
            train_df['Path'] = os.path.join(input_dir, "images/") + train_df.img_name
            cv = StratifiedGroupKFold(n_splits=5, shuffle = True, random_state=42)
            train_sets = []
            val_sets = []
            ## Create 5 CV-Folds in a grouped and stratified fashion (we dont want data leaks)
            for fold , (train_inds, test_inds) in enumerate(cv.split(X=train_df, y=train_df.label, groups=train_df.img_name)):
                train_df_cv = train_df.iloc[train_inds]
                val_df_cv = train_df.iloc[test_inds]
                train_df_cv.to_csv(output_dir+ f'/nodule_train_fold{fold}.csv')
                val_df_cv.to_csv(output_dir + f'/nodule_val_fold{fold}.csv')
                if yolo:
                    if not os.path.exists(f"{output_dir}/nodule_train_fold{fold}.txt"):
                        f=open(f"{output_dir}/nodule_train_fold{fold}.txt", "a+")
                        for idx, row in train_df_cv.iterrows():
                            name = row.Path
                            f.write(f"{name}\n")
                        f.close()
            
                        f=open(f"{output_dir}/nodule_val_fold{fold}.txt", "a+")
                        for idx, row in val_df_cv.iterrows():
                            name = row.Path
                            f.write(f"{name}\n")
                        f.close()

            if yolo: 
                # only needed once
                if not os.path.exists(f'{input_dir}/labels'):
                    os.makedirs(f'{input_dir}/labels',exist_ok=True)
                    # convert data to yolo format ({x_center} {y_center} {width} {height}) ene .txt file per image
                if len(os.listdir(f'{input_dir}/labels')) == 0:
                    df = train_df[train_df['label']==1]

                    df.x = (df.x + df.width / 2) / 1024
                    df.y = (df.y + df.height / 2) / 1024
                    df.height = df.height / 1024
                    df.width = df.width / 1024
                    for idx, row in df.iterrows():
                        name = row.img_name.replace('.mha','')
                        label = row.label - 1
                        x_center = row.x
                        y_center = row.y
                        width = row.width
                        height = row.height
                        f=open(f"{input_dir}/labels/{name}.txt", "a+")
                        f.write(f"{label} {x_center} {y_center} {width} {height}\n")
                        f.close()

                for fold in range(5): # train and validate each Fold
                    opt = self.parse_opt() # initial params
                    if '_l' in modelname: # larger input (1024px)
                        opt.hyp = '/opt/algorithm/training_utils/yolov5/hyp_1024.yaml' if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/training_utils/yolov5/hyp_1024.yaml"
                        opt.img = 1024
                    else : # smaller input (640px)
                        opt.hyp = '/opt/algorithm/training_utils/yolov5/hyp_640.yaml' if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/training_utils/yolov5/hyp_640.yaml"
                        opt.img = 640
                    # Write data info to yaml (bit of a hack)
                    data = {'path': str(input_dir) ,
                            'train': f"{output_dir}/nodule_train_fold{fold}.txt", 
                            'val': f"{output_dir}/nodule_val_fold{fold}.txt",
                            'nc': 1 , 
                            'names': ['nodule']}
                    with open(f'{output_dir}/node_dataset_fold{fold}.yaml', 'w') as outfile:
                        yaml.dump(data, outfile, default_flow_style=False)
                    # specify training parameters
                    opt.device = 0
                    opt.data = f'{output_dir}/node_dataset_fold{fold}.yaml'
                    opt.epochs=20
                    opt.savedir = model_ckpt_dir
                    opt.project = model_ckpt_dir
                    opt.name = ''
                    opt.prefix = f'yolo-fold-{fold+1}-'
                    if self.retrain:
                        opt.weights = self.archs[modelname][f'fold-{fold+1}'][0] 
                    else: 
                        opt.weights = '/opt/algorithm/yolo5x_vindr.pt' if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/yolo5x_vindr.pt"
                    # check if the model is already trained
                    if os.path.exists(model_ckpt_dir + f'/yolo-fold-{fold+1}-last.pt'): 
                        ckpt = torch.load(model_ckpt_dir + f'/yolo-fold-{fold+1}-last.pt')
                        start_epoch = ckpt['epoch'] + 1
                        if start_epoch >= opt.epochs or start_epoch == 0  :
                            continue
                        del ckpt
                        opt.resume = model_ckpt_dir + f'/yolo-fold-{fold+1}-last.pt'
                    else: 
                        opt.resume=False
                    # train
                    model_path = train_yolo(opt)


            else: 
                # Pytorch lightning pipeline
                # load config (we need one config file for each model with all important training params etc.)
                configpath = Path(f'/opt/algorithm/config_{modelname}.yaml') if execute_in_docker else Path(f"//home/Behrendt/projects/Node21/node21-submit/config_{modelname}.yaml/")
                cfg = OmegaConf.load(configpath)
                base = cfg.callbacks.model_checkpoint.monitor 
                for fold in range(5): # iterate over folds 
                    prefix = f'{fold+1}/'
                    cfg.datamodule.cfg.path.train.images = ''
                    cfg.datamodule.cfg.path.val.images = ''
                    cfg.datamodule.cfg.path.train.labels = [output_dir+ f'/nodule_train_fold{i}.csv' for i in range(5)]
                    cfg.datamodule.cfg.path.val.labels = [output_dir+ f'/nodule_val_fold{i}.csv' for i in range(5)]
                    
                    # Init Data Module
                    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule,fold = fold)
                    if 'fcrnn' in modelname:
                        cfg.model.cfg.vindr_path = '/opt/algorithm/fastercnn50.pth' if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/fastercnn50.pth"
                    elif 'effdet' in modelname :
                        cfg.model.cfg.vindr_path = '/opt/algorithm/F1_E79_ModelX_v4_T0.325_V0.410.ckpt' if execute_in_docker else "//home/Behrendt/projects/Node21/node21-submit/F1_E79_ModelX_v4_T0.325_V0.410.ckpt"
                   
                    # Init lightning model
                    model: LightningModule = hydra.utils.instantiate(cfg.model, prefix=prefix) # what about passing cfg.datamodule here? This would also avoid e.g. the mapping of num_classes 
                
                    # Init lightning callbacks
                    cfg.callbacks.model_checkpoint.monitor = f'{prefix}' + base
                    cfg.callbacks.model_checkpoint.filename = "epoch-{epoch}_step-{step}_loss-{"+f"{prefix}"+"val/losses/loss:.2f}"
                    cfg.callbacks.model_checkpoint.dirpath = model_ckpt_dir
                   
                    if self.retrain:
                        cfg.trainer.resume_from_checkpoint = str(self.archs[modelname][f'fold-{fold+1}'][0])
                    else:
                        cfg.trainer.resume_from_checkpoint = None
                    # Resume training. The trainer will skip training if complete
                    if os.path.exists(model_ckpt_dir + f'/last_{modelname}_fold-{fold+1}.ckpt'): 
                        cfg.trainer.resume_from_checkpoint = model_ckpt_dir + f'/last_{modelname}_fold-{fold+1}.ckpt'
                    else:
                        cfg.trainer.resume_from_checkpoint = None

                    callbacks: List[Callback] = []
                    if "callbacks" in cfg:
                        for _, cb_conf in cfg.callbacks.items():
                            if "_target_" in cb_conf:
                                callbacks.append(hydra.utils.instantiate(cb_conf))

                        callbacks[0].FILE_EXTENSION = f'_{modelname}_fold-{fold+1}.ckpt' 

                    # Init Lightning Trainer
                    # cfg.trainer.max_epochs=1 
                    trainer: Trainer = hydra.utils.instantiate(
                        cfg.trainer, callbacks=callbacks, logger=None, _convert_="partial", plugins=None
                    )
                    trainer.fit(model, datamodule)

      

    def format_to_GC(self, np_prediction, spacing) -> Dict:
        '''
        Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation
        
        
        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.
        
        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters. 
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
        x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
        boxes = []
        for i, bb in enumerate(np_prediction['boxes']):
            box = {}   
            box['corners']=[]
            x_min, y_min, x_max, y_max = bb*x_y_spacing
            x_min, y_min, x_max, y_max  = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
            bottom_left = [x_min, y_min,  np_prediction['slice'][i]] 
            bottom_right = [x_max, y_min,  np_prediction['slice'][i]]
            top_left = [x_min, y_max,  np_prediction['slice'][i]]
            top_right = [x_max, y_max,  np_prediction['slice'][i]]
            box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
            box['probability'] = round(float(np_prediction['scores'][i]), 2)
            boxes.append(box)
        
        return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={ "major": 1, "minor": 0 })
        
    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d

    def _create_dummy_inference_targets(self,num_images,device,img_size):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(img_size, img_size)] * num_images, device=device
            ).float(),
            "img_scale": torch.ones(num_images, device=device).float(),
        }

        return dummy_targets   

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        indexes = torch.where(scores > 0.01)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        agg_results = []
        agg_results_per_model = []
        for arch in self.models:
            for fold in self.models[arch]:
                results = []
                for model in self.models[arch][fold]:
                    self.effdet = False 
                    self.yolo = False
                    self.detr = False
                    if '_s' in str(arch).lower():
                        input_res = 512
                        yolores = 640
                        aug = True
                    else:
                        input_res = 1024
                        yolores = 1024
                        aug = False
                    if 'effdet' in str(arch).lower():
                        self.effdet = True
                        # input_res = model._hparams.cfg.new_shape
                        dummy_targets = self._create_dummy_inference_targets(1,self.device,input_res)
                    elif 'detr' in str(arch).lower():
                        self.detr = True
                    elif 'yolo' in str(arch).lower():
                        self.yolo = True
                    model.eval()
                    # input_image = SimpleITK.ReadImage("/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/images/n0163.mha")
                    image_data = SimpleITK.GetArrayFromImage(input_image)
                    spacing = input_image.GetSpacing()
                    image_data = np.array(image_data)
                    
                    if len(image_data.shape)==2:
                        image_data = np.expand_dims(image_data, 0)
                        
                    
                    print('pred')
                    # operate on 3D image (CXRs are stacked together)
                    for j in range(len(image_data)):
                        # Pre-process the image
                        image = image_data[j,:,:]
                        # The range should be from 0 to 1.
                        
                        if input_res != 1024 and not self.yolo:
                            print('changing resolution')
                            old_size = 1024
                            image = A.Resize(input_res,input_res)(image=image)['image']
                        image = image.astype(np.float32) / np.max(image)  # normalize
                        image = np.expand_dims(image, axis=0)
                        tensor_image = torch.from_numpy(image).to(self.device)#.reshape(1, 1024, 1024)
                        with torch.no_grad():         
                            if self.effdet:
                                input = torch.stack([tensor_image,tensor_image,tensor_image],1) # to rgb
                                prediction = model(input,dummy_targets)['detections'][0]
                                postprocessed = self._postprocess_single_prediction_detections(prediction) 
                                boxes = postprocessed['boxes']
                                if input_res != 1024: # rescale bbox predictions to the original size
                                    boxes = boxes * old_size / input_res
                                scores = postprocessed['scores']
                                prediction = [{'boxes':boxes , 'scores':scores }]
                                prediction = [get_NonMaxSup_boxes(prediction[0])]

                            elif self.detr:
                                input = torch.stack([tensor_image,tensor_image,tensor_image],1)
                                pred = model(input)
                                orig_target_sizes = torch.tensor([[input_res, input_res]],device=self.device)
                                pred = self.postprocessors(pred, orig_target_sizes)
                                boxes = []
                                scores = []
                                for box, score in zip(pred[0]['boxes'],pred[0]['scores']): 
                                    if score > 0.1:
                                        if input_res != 1024:
                                            box = box * 1024 / input_res
                                        boxes.append(box)
                                        scores.append(score)
                                prediction = [{'boxes':boxes , 'scores':scores}]
                                prediction = [get_NonMaxSup_boxes(prediction[0])]

                            elif self.yolo:
                                input = torch.stack([tensor_image,tensor_image,tensor_image],1)
                                input = torchvision.transforms.ToPILImage()(input.squeeze())
                                pred = model(input,size=yolores,augment=True)

                                if len(pred.xyxy[0])!=0:
                                    prediction = [{'boxes': [x[0:4] for x in pred.xyxy[0]], 'scores': [x[4] for x in pred.xyxy[0]]}]
                                else:
                                    prediction = [{'boxes': torch.empty([0,4], dtype=torch.float32), 'scores': torch.tensor([])}] # to be changed?
                                prediction = [get_NonMaxSup_boxes(prediction[0])]
                            else :  
                                prediction = model([tensor_image.to(self.device)])
                                if input_res != 1024:
                                    prediction[0]['boxes'] = prediction[0]['boxes'] * 1024 / input_res 
                                prediction = [get_NonMaxSup_boxes(prediction[0])]

                        # convert predictions from tensor to numpy array.
                        np_prediction = {str(key):[i.cpu().numpy() for i in val]
                            for key, val in prediction[0].items()}

                        np_prediction['slice'] = len(np_prediction['boxes'])*[j]
                        results.append(np_prediction)
                agg_results.extend([results])

            if self.ensemble_per_model:
                agg_results_per_model.extend([ensemble_boxes(agg_results,skip_boxes_thresh=0.1)])

        if self.ensemble_per_model:
            ens_results = ensemble_boxes(agg_results_per_model,skip_boxes_thresh=0.1) 
        else:
            ens_results = ensemble_boxes(agg_results,skip_boxes_thresh=0.1) 

        # [x.pop('labels') for x in ens_results]
        predictions = self.merge_dict(ens_results)
        data = self.format_to_GC(predictions, spacing)
        print(data)
        return data

    def parse_opt(known=False): # default yolo params
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
        parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok',default=True, help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=12, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

        # Weights & Biases arguments
        parser.add_argument('--entity', default=None, help='W&B: Entity')
        parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
        parser.add_argument('--bbox_interval', type=int, default=5, help='W&B: Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

        opt = parser.parse_known_args()[0] if known else parser.parse_args()
        return opt
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    parser.add_argument('--train', action='store_true', help = "Algorithm on train mode.")
    parser.add_argument('--retrain', action='store_true', help = "Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest', action='store_true', help = "Algorithm on evaluate mode after retraining.")

    parsed_args = parser.parse_args()  
    if (parsed_args.train or parsed_args.retrain):# train mode: retrain or train
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, parsed_args.train, parsed_args.retrain, parsed_args.retest).train()
    else:# test mode (test or retest)
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, retest=parsed_args.retest).process()
            
    
   
    
    
    
    
    
