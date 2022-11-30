'''
Text format: [Id]###[Title]###[Cont]###http:xxx.jpg###xyz<tag>xxx<tag>xxx...
We only use the [Cont] part, if it is not null and has a valid accompanying image.
'''

import os
from PIL import Image

img_dir = './MASAD_imgs'
imgs = os.listdir(img_dir)


def f(fin_name):
    if 'train' in fin_name:
        fout_name = 'train_filtered.tsv'
    elif 'test' in fin_name:
        fout_name = 'test.tsv'
    else:
        raise RuntimeError('Illegal root path for MASAD')
    fin = open(fin_name, 'r', encoding='utf-8')
    fout = open(fout_name, 'w', encoding='utf-8')

    lines = fin.readlines()
    fout.write(lines.pop(0))

    empty_cont_num = 0
    for line in lines:
        line = line.split('\t')
        idx = line[0]
        polarity = line[1]
        img = line[2]
        if img not in imgs:
            print(f'[#] img not found: {img}')
            continue

        try:
            Image.open(os.path.join(img_dir, img)).convert('RGB')
        except:
            print(f"[#] img broken, open failed: {img}")
            continue

        cont = line[3]
        aspect = line[4]  # including \n

        cont = cont.split('###')[2]
        if cont == '':
            empty_cont_num += 1
            continue

        cont = cont.replace('\(', '(')
        cont = cont.replace('\)', ')')
        cont = cont.replace('\?', '?')
        fout.write(f'{idx}\t{polarity}\t{img}\t{cont}\t{aspect}')
    print('[#] Empty cont num:', empty_cont_num)
    fin.close()
    fout.close()


f('train_raw.tsv')
f('test_raw.tsv')
