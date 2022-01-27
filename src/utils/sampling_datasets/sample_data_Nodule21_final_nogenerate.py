import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, StratifiedGroupKFold



#
# get 100 samples from simulatied_metadata.csv to generate nodules in these images
# then we build a 50/50 training set of nodules/non-nodules
#
path_data_train = '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/metadata.csv'
imgpath = '/proccessed_data/images/'
savePath =  '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/splits/'
mappingpath = '/home/Behrendt/data/LUMEN/Node21/cxr_images/original_data/filenames_orig_and_new.csv' 
mappingpath2 = '/home/Behrendt/data/LUMEN/Node21/cxr_images/original_data/non_nodule_filenames_orig_and_new.csv' 
# simulated_path = '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/simulated_metadata.csv'
image_path = '/proccessed_data/images/'
# basepath_plots = r'C:\Users\Finn\Documents\Projects\LUMEN\Dataset investigation\plots\Chexpert/'

# df_sim = pd.read_csv(simulated_path)

df = pd.read_csv(path_data_train)
# df['Path'] = imgpath + df.img_name
mapping = pd.read_csv(mappingpath2)
mapping_nodule = pd.read_csv(mappingpath)
# JSRT -- 149 nodules and 93 non nodule images  -- we only can get the nodule data...
# jrst_h = mapping[mapping.orig_dataset=='jsrt']
# jrst_u = mapping_nodule[mapping_nodule.orig_dataset=='jsrt']
# test_df = df.loc[df.img_name.str.replace('.mha','').isin(jrst_h.node21_img_id.values)].sample(50,random_state=44)
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(jrst_u.node21_img_id.values)].sample(50,random_state=44))
# # OpenI 1102/54
# openi_h = mapping[mapping.orig_dataset=='openi']
# openi_u = mapping_nodule[mapping_nodule.orig_dataset=='openi']
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(openi_h.node21_img_id.values)].sample(25,random_state=40))
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(openi_u.node21_img_id.values)].sample(25,random_state=40))
# # CestXray14  617 / 1187
# cxr_h = mapping[mapping.orig_dataset=='chestxray14']
# cxr_u = mapping_nodule[mapping_nodule.orig_dataset=='chestxray14']
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(cxr_h.node21_img_id.values)].sample(50,random_state=44))
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(cxr_u.node21_img_id.values)].sample(50,random_state=44))
# # padchest 314 / 1366
# pc_h = mapping[mapping.orig_dataset=='padchest']
# pc_u = mapping_nodule[mapping_nodule.orig_dataset=='padchest']
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(pc_h.node21_img_id.values)].sample(50,random_state=45))
# test_df = test_df.append(df.loc[df.img_name.str.replace('.mha','').isin(pc_u.node21_img_id.values)].sample(50,random_state=45))


# sanity check
# print(len(test_df.img_name.unique()))
# unique_list =test_df.img_name.unique() 

# train_df = df.loc[~df.img_name.isin(test_df.img_name)]
train_df = df
# df_sim['Path'] = generated_path + df_sim.img_name
train_df['Path'] = image_path + train_df.img_name
# generate_df = df_sim.loc[~df_sim.img_name.isin(test_df.img_name)]
# generate_df = df_sim.loc[~generate_df.img_name.isin(df.img_name)]
# for i, row in train_df.iterrows():
#     if row.img_name in unique_list:
#         print('warning ')


#
# generate_df = generate_df.sample(1400,random_state=57)
# generate_unique = generate_df.groupby('img_name').apply(lambda df: df.sample(1)).sample(1200, random_state=42)

# generate_df = generate_df.loc[generate_df.img_name.isin(generate_unique.img_name)]
# add_nodules = generate_df.loc[~generate_df['Unnamed: 0'].index.isin(generate_unique['Unnamed: 0'])].sample(400, random_state=42)
# generate_df = generate_df.append(add_nodules)
# generate_df.to_csv(savePath+f'nodules_to_generate.csv')
# generate_df = pd.read_csv(savePath+f'nodules_to_generate.csv')
# train_df = train_df.loc[~train_df.img_name.isin(generate_df.img_name)]
# train_df = train_df.append(generate_df)
# train_df = train_df.fillna(1.0) 
# Split to train/val and Test Data with group awareness and View stratification
cv = StratifiedGroupKFold(n_splits=5,shuffle = True, random_state=42)
# for fold, (train_inds, test_inds) in enumerate(cv.split(X=df, y=df.label, groups=df.img_name)):
# # train_inds, test_inds = next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 42).split(df, groups=df['patient']))
#     if fold == 0:
#         train_df = df.iloc[train_inds]
#         test_df = df.iloc[test_inds]

# test_df.to_csv(savePath+f'nodule_test_v3.csv')

for fold , (train_inds, test_inds) in enumerate(cv.split(X=train_df, y=train_df.label, groups=train_df.img_name)):
    train_df_cv = train_df.iloc[train_inds]
    val_df_cv = train_df.iloc[test_inds]

    train_df_cv.to_csv(savePath+f'nodule_train_fold{fold}_finalnogen.csv')
    val_df_cv.to_csv(savePath+f'nodule_val_fold{fold}_finalnogen.csv')



print(f'Length of Training Set(s): {len(train_df_cv)}')
print(f'Length of Validation Set(s): {len(val_df_cv)}')
# print(f'Length of Test Set: {len(test_df)}')
print('done')