# Please excute following commands first
# $ unzip IJCAI2019_data.zip
# $ cp IJCAI2019_data/twitter2017/train.tsv ./
# $ cp IJCAI2019_data/twitter2017/dev.tsv ./
# $ cp IJCAI2019_data/twitter2017/test.tsv ./

for fin_name, fout_name in zip(['train.tsv', 'dev.tsv'], ['train_sort.tsv', 'dev_sort.tsv']):
    fin = open(fin_name, 'r', encoding='utf-8')
    fout = open(fout_name, 'w', encoding='utf-8')

    lines = fin.readlines()
    fout.write(lines.pop(0))
    lines.sort(key=lambda x: x.split('\t')[1])

    for line in lines:
        fout.write(line)

    fin.close()
    fout.close()
