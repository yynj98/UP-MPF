import random
random.seed(34)

fin_name = 'train_sort.tsv'
fout1_name = 'train_few1.tsv'
fout2_name = 'train_few2.tsv'

cata_num = 3
cata_start_idx = [1, 369, 2252, 3179]  # the last row of each category
k = 12

fin = open(fin_name, 'r', encoding='utf-8')
fout1 = open(fout1_name, 'w', encoding='utf-8')
fout2 = open(fout2_name, 'w', encoding='utf-8')

lines = fin.readlines()
fout1.write(lines[0])
fout2.write(lines[0])

for i in range(cata_num):
    sample_range = range(cata_start_idx[i], cata_start_idx[i+1])
    sample_idx = random.sample(sample_range, k=2*k)
    # sample_idx.sort()
    for idx in sorted(sample_idx[:k]):
        fout1.write(lines[idx])
    for idx in sorted(sample_idx[k:]):
        fout2.write(lines[idx])

fin.close()
fout1.close()
fout2.close()
