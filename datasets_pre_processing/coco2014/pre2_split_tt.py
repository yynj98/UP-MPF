'''
Split train.txt and test.txt from trainval_raw.txt by a ratio of 9:1 in UP stage
'''

import random
random.seed(34)

fin = open('trainval_raw.txt', 'r', encoding='utf-8')
ftrain = open('train.txt', 'w', encoding='utf-8')
ftest = open('test.txt', 'w', encoding='utf-8')

lines = fin.readlines()
l = list(range(len(lines)))
random.shuffle(l)

train_range = len(l) // 10 * 9
for idx in sorted(l[:train_range]):
    ftrain.write(lines[idx])

test_indices = l[train_range:]
keep_range = len(test_indices) // 3
left_range = len(test_indices) // 3 * 2

for idx in sorted(test_indices[:keep_range]):
    ftest.write('{}\tkeep\n'.format(lines[idx].rstrip()))
for idx in sorted(test_indices[keep_range:left_range]):
    ftest.write('{}\tleft\n'.format(lines[idx].rstrip()))
for idx in sorted(test_indices[left_range:]):
    ftest.write('{}\tright\n'.format(lines[idx].rstrip()))

fin.close()
ftrain.close()
ftest.close()
