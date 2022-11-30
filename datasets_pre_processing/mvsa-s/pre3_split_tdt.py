'''
Split train.tsv, dev.tsv and test.tsv from all.tsv by a ratio of 8:1:1
'''

import random
random.seed(34)

fin = open('all.tsv', 'r', encoding='utf-8')
ftrain = open('train.tsv', 'w', encoding='utf-8')
fdev = open('dev.tsv', 'w', encoding='utf-8')
ftest = open('test.tsv', 'w', encoding='utf-8')

lines = fin.readlines()
title = lines.pop(0)
ftrain.write(title)
fdev.write(title)
ftest.write(title)

l = list(range(len(lines)))
random.shuffle(l)

train_range = len(l) // 10 * 8
dev_range = len(l) // 10 * 1

for idx in sorted(l[:train_range]):
    ftrain.write(lines[idx])

for idx in sorted(l[train_range: train_range + dev_range]):
    fdev.write(lines[idx])

for idx in sorted(l[train_range + dev_range: ]):
    ftest.write(lines[idx])

fin.close()
ftrain.close()
fdev.close()
ftest.close()
