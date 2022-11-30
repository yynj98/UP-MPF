# Please excute following commands first
# $ unzip train2014.zip
# $ unzip val2014.zip

import os
from shutil import copy

trainval_dir = 'coco_trainval2014'
os.makedirs(trainval_dir)

train_dir = 'train2014'
train_imgs = os.listdir(train_dir)
print(len(train_imgs))
with open('train_raw.txt', 'w', encoding='utf-8') as f:
    for img in train_imgs:
        f.write(f'{img}\n')
        copy(os.path.join(train_dir, img), trainval_dir)

val_dir = 'val2014'
val_imgs = os.listdir(val_dir)
print(len(val_imgs))
with open('val_raw.txt', 'w', encoding='utf-8') as f:
    for img in val_imgs:
        f.write(f'{img}\n')
        copy(os.path.join(val_dir, img), trainval_dir)

with open('trainval_raw.txt', 'w', encoding='utf-8') as f:
    for img in train_imgs:
        f.write(f'{img}\n')
    for img in val_imgs:
        f.write(f'{img}\n')
