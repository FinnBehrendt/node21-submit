import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold

path_data_train = '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/metadata.csv'
imgpath = '/proccessed_data/images/'
savePath =  '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/splits/'
mappingpath = '/home/Behrendt/data/LUMEN/Node21/cxr_images/original_data/filenames_orig_and_new.csv' 
mappingpath2 = '/home/Behrendt/data/LUMEN/Node21/cxr_images/original_data/non_nodule_filenames_orig_and_new.csv' 
# basepath_plots = r'C:\Users\Finn\Documents\Projects\LUMEN\Dataset investigation\plots\Chexpert/'
df = pd.read_csv(path_data_train)
df['Path'] = imgpath + df.img_name
mapping = pd.read_csv(mappingpath)
mapping2 = pd.read_csv(mappingpath2)
# JSRT -- 149 nodules and 93 non nodule images  -- we only can get the nodule data...
mapping = mapping[mapping.orig_dataset=='jsrt']
mapping2 = mapping2[mapping2.orig_dataset=='jsrt']

test_df = df.loc[df.img_name.str.replace('.mha','').isin(mapping.node21_img_id.values)]

healthy = df.loc[df.img_name.str.replace('.mha','').isin(mapping2.node21_img_id.values)]
test_df = test_df.append(healthy)

# sanity check
print(len(healthy.img_name.unique()))

train_df = df.loc[~df.img_name.isin(test_df.img_name)]

# Split to train/val and Test Data with group awareness and View stratification
cv = StratifiedGroupKFold(n_splits=5,shuffle = True, random_state=42)
# for fold, (train_inds, test_inds) in enumerate(cv.split(X=df, y=df.label, groups=df.img_name)):
# # train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42).split(df, groups=df['patient']))
#     if fold == 0:
#         train_df = df.iloc[train_inds]
#         test_df = df.iloc[test_inds]

# test_df.to_csv(savePath+f'nodule_test.csv')

for fold , (train_inds, test_inds) in enumerate(cv.split(X=train_df, y=train_df.label, groups=train_df.img_name)):
    train_df_cv = train_df.iloc[train_inds]
    val_df_cv = train_df.iloc[test_inds]

    # train_df_cv.to_csv(savePath+f'nodule_train_fold{fold}.csv')
    # val_df_cv.to_csv(savePath+f'nodule_val_fold{fold}.csv')



print(f'Length of Training Set(s): {len(train_df_cv)}')
print(f'Length of Validation Set(s): {len(val_df_cv)}')
print(f'Length of Test Set: {len(test_df)}')
print('done')