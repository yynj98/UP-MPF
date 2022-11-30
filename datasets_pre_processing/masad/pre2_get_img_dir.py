'''
Copy the original classified pictures to one directory
'''

import os
import os.path as op
from shutil import copy

img_dir = './MASAD_imgs'
os.mkdir(img_dir)

imgs_path = []
imgs_set = set()
root = './MASAD_data'
for root in [op.join(root, 'train', 'image'), op.join(root, 'test', 'image')]:
    for polarity in ['negative', 'positive']:
        path1 = op.join(root, polarity)
        for cata in os.listdir(path1):
            path2 = op.join(path1, cata)
            if op.isdir(path2):
                for img in os.listdir(path2):
                    if img in imgs_set:  # there may be same images in different directories
                        continue
                    else:
                        imgs_set.add(img)
                        path3 = op.join(path2, img)  # full image path
                        imgs_path.append(path3)
print(len(imgs_path))
for img in imgs_path:
    copy(img, img_dir)
