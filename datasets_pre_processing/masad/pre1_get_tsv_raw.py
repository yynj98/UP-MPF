# Please excute the following command first
# $ unzip MASAD.zip -d MASAD_data

import os
import os.path as op


def f(root):
    if 'train' in root:
        fout_name = 'train_raw.tsv'
    elif 'test' in root:
        fout_name = 'test_raw.tsv'
    else:
        raise RuntimeError('Illegal root path for MASAD')
    fout = open(fout_name, 'w', encoding='utf-8')
    fout.write('index	#1 Label	#2 ImageID	#3 String	#3 String\n')

    for polarity in ['negative', 'positive']:
        path1 = op.join(root, polarity)
        for cata in os.listdir(path1):
            path2 = op.join(path1, cata)
            print(path2)
            if op.isdir(path2):
                aspect = cata
                # print(aspect)
                for txt in os.listdir(path2):
                    idx, _ = op.splitext(txt)
                    with open(op.join(path2, txt), 'r', encoding='utf-8') as f:
                        l = f.readlines()
                        if len(l) != 1:
                            print(op.join(path2, txt))
                            continue
                        cont = l[0]
                        fout.write(f'{idx}\t{polarity}\t{idx}.jpg\t{cont}\t{aspect}\n')
    fout.close()


f('MASAD_data/train/text')
f('MASAD_data/test/text')
