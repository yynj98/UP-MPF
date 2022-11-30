'''
Split train.tsv and dev.tsv from train_filtered.tsv by a ratio of 9:1
'''

import random
random.seed(34)

fin = open('train_filtered.tsv', 'r', encoding='utf-8')
ftrain = open('train.tsv', 'w', encoding='utf-8')
fdev = open('dev.tsv', 'w', encoding='utf-8')

lines = fin.readlines()
title = lines.pop(0)
ftrain.write(title)
fdev.write(title)

l = list(range(len(lines)))
random.shuffle(l)

train_range = len(l) // 10 * 9

for idx in sorted(l[:train_range]):
    ftrain.write(lines[idx])

for idx in sorted(l[train_range: ]):
    fdev.write(lines[idx])

fin.close()
ftrain.close()
fdev.close()
