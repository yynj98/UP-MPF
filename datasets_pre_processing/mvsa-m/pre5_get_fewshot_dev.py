import random
random.seed(34)

fin_name = 'dev.tsv'
fout1_name = 'dev_few1.tsv'
fout2_name = 'dev_few2.tsv'

cata_num = 3
cata_start_idx = [1, 127, 532, 1614]  # the last row of each category
k = 43  # 12904 * 0.01 = 129 = 43 * 3

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
