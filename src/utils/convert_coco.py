import pandas as pd 
# convert the metatdata into COCO format, i.e. .txt files for every sample with nodules.

basepath = '/home/Behrendt/projects/yolo/datasets/node/'
# images_path = basepath + 'images/'
# labels = '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/metadata.csv'
labels = '/home/Behrendt/data/LUMEN/Node21/cxr_images/proccessed_data/simulated_nodules_v4.csv'
# labels_path = basepath + 'labels/'
labels_path = basepath + 'labels_generated_nodules/'
df = pd.read_csv(labels)


df = df[df['label']==1]

df.x = (df.x + df.width / 2) / 1024
df.y = (df.y + df.height / 2) / 1024
df.height = df.height / 1024
df.width = df.width / 1024

# txt file: for each image with label=1 one .txt file. with format: class x_center y_center width height 

for idx, row in df.iterrows():
    name = row.img_name.replace('.mha','')
    label = row.label - 1
    x_center = row.x
    y_center = row.y
    width = row.width
    height = row.height
    f=open(f"{labels_path}{name}.txt", "a+")
    f.write(f"{label} {x_center} {y_center} {width} {height}\n")
    f.close()
print('done')

