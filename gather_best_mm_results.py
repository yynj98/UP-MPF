import os
import os.path as op

fout = open('best_mm_results.txt', 'w', encoding='utf-8')
root = 'out'
datasets = ['mvsa-s', 'mvsa-s+']  # Options: ['t2015', 't2015+', ... ,'tumemo', 'tumemo+']
for dataset in datasets:
    fout.write(dataset + '\n')
    dir_1 = op.join(root, dataset)
    templates = os.listdir(dir_1)
    templates.sort(key=lambda x: (x[:12], x[-6:]))  # [s1/2][d1/2][t1/2], [lp11/22/.../77]
    for template in templates:
        if template[-6:] not in ['[lp11]', '[lp22]', '[lp33]', '[lp44]', '[lp55]', '[lp66]', '[lp77]']:
            continue
        fout.write(template + '\n')
        dir_2 = op.join(dir_1, template)

        if '+' not in dataset:
            files = os.listdir(dir_2)
            files = [f for f in files if f[-4:] == '.txt']
            for file in files:  # only one file here actually
                fout.write(file[:-4] + '\n')
            continue

        loads = os.listdir(dir_2)
        loads.sort()
        result_files = []
        for load in loads:
            dir_3 = op.join(dir_2, load)
            files = os.listdir(dir_3)
            files = [f for f in files if f[-4:] == '.txt']
            result_files += files

        idx = 0
        best_idx = 0
        best_value = 0
        for f in result_files:
            values = f.split('_')[4:7]  # mean values (Acc, Mac-F1, Wtd-F1)
            value = sum([float(i) for i in values])
            if value > best_value:
                best_value = value
                best_idx = idx
            idx += 1
        fout.write(result_files[best_idx][:-4] + '\n')
    fout.write('\n\n')
fout.close()
