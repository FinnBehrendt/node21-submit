import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from utils import *
from typing import Dict
import json
from skimage.measure import regionprops
import imageio
from pathlib import Path
import time
import pandas as pd
import random
from random import randrange
import os
from tqdm import tqdm 
# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = False
class Nodulegeneration(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/") if execute_in_docker else Path("/home/linux/Node21/generation_task//test/"),
            output_path = Path("/output/") if execute_in_docker else Path("/home/linux/Node21/generation_task//output/"),
            output_file = Path("/output/results.json") if execute_in_docker else Path("./output/results.json")

        )

        # load nodules.json for location
        # with open("/input/nodules.json" if execute_in_docker else "/home/linux/Node21/generation_task/test/nodules.json") as f:
        #     self.data = json.load(f)
        self.data = pd.read_csv('/home/linux/Node21/generation_task/nodules_to_generate.csv')


    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape)==2:
            input_image = np.expand_dims(input_image, 0)
        
        pd_data = pd.read_csv('/opt/algorithm/ct_nodules.csv' if execute_in_docker else "/home/linux/Node21/generation_task/ct_nodules.csv")
        data_path = '/mnt/data/Behrendt/Node21/cxr_images/proccessed_data/images'
        save_path = '/mnt/data/Behrendt/Node21/cxr_images/proccessed_data/images_generated/'
        nodule_images = np.zeros(input_image.shape)
        img_names = self.data.img_name.tolist()
        for img in tqdm(img_names):
            t = time.time()
            path = os.path.join(Path(data_path),Path(img))
            input = SimpleITK.ReadImage(path)
            input_image = SimpleITK.GetArrayFromImage(input)
            cxr_img_scaled = input_image
            # nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2]==j]
            nodule_data = self.data[self.data.img_name==img]

            for i, nodule in nodule_data.iterrows():
                # print(len(nodule))
                cxr_img_scaled = convert_to_range_0_1(cxr_img_scaled)
                # boxes = nodule['corners']
                # no spacing info in GC with 3D version
                #x_min, y_min, x_max, y_max = boxes[2][0]/spacing_x, boxes[2][1]/spacing_y, boxes[0][0]/spacing_x, boxes[0][1]/spacing_y
                y_min, x_min, y_max, x_max = nodule.x, nodule.y, nodule.x + nodule.width, nodule.y + nodule.height

                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                #------------------------------ Randomly choose ct patch and scale it according to bounding box size.
                required_diameter = max(x_max-x_min, y_max-y_min)
                ct_names = pd_data[pd_data['diameter']>int((required_diameter/5))]['img_name'].values
                if len(ct_names)<1:
                    ct_names = pd_data[pd_data['diameter']>int((required_diameter/10))]['img_name'].values
                    
                index_ct = random.randint(0, len(ct_names)-1)
                path_nodule = '/opt/algorithm/nodule_patches/' if execute_in_docker else '/home/linux/Node21/generation_task/nodule_patches/'
                X_ct_2d_resampled, diameter = process_CT_patches(os.path.join(path_nodule,ct_names[index_ct]), os.path.join(path_nodule, ct_names[index_ct].replace('dcm','seg')), required_diameter)
                
                crop = cxr_img_scaled[x_min:x_max, y_min:y_max].copy()
                new_arr = convert_to_range_0_1(X_ct_2d_resampled)

                # contrast matching:
                c = contrast_matching(new_arr, cxr_img_scaled[x_min:x_max, y_min:y_max])
                nodule_contrasted = new_arr * c

                indexes = nodule_contrasted!=np.min(nodule_contrasted)
                result = poisson_blend(nodule_contrasted, cxr_img_scaled, y_min, y_max, x_min, x_max)
                result[x_min:x_max, y_min:y_max] = np.mean(np.array([crop*255, result[x_min:x_max, y_min:y_max]]), axis=0)
                cxr_img_scaled = result.copy()

            final_image = SimpleITK.GetImageFromArray(result)
            SimpleITK.WriteImage(final_image,os.path.join(save_path,img))
            # nodule_images[j,:,:] = result 
        print('total time took ', time.time()-total_time)
       # return SimpleITK.GetImageFromArray(nodule_images)

if __name__ == "__main__":
    Nodulegeneration().process()

                                                                 
                                                                 
