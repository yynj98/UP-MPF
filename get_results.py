import os
import re
import torch
import os.path as op

p = re.compile(r'\[Test(.+?)-(.+?)-(.+?)\]')
lr_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]

out_dir = 'out'
dataset = 'mvsa-s'      # Options: 't2015', 't2017', 'masad', 'mvsa-s', 'mvsa-m', 'tumemo'
train_dev = '[s1][d1]'  # Options: '[s1][d1]', '[s2][d2]'

ph = ''
if dataset[:5] in ['t2015', 't2017', 'masad']:
    ph = '1'

templates = [
    f'[t1][ps111{ph}][nf_resnet50][lp11]',
    f'[t1][ps111{ph}][nf_resnet50][lp22]',
    f'[t1][ps111{ph}][nf_resnet50][lp33]',
    f'[t1][ps111{ph}][nf_resnet50][lp44]',
    f'[t1][ps111{ph}][nf_resnet50][lp55]',
    f'[t1][ps111{ph}][nf_resnet50][lp66]',
    f'[t1][ps111{ph}][nf_resnet50][lp77]',

    '[t2][nf_resnet50][lp11]',
    '[t2][nf_resnet50][lp22]',
    '[t2][nf_resnet50][lp33]',
    '[t2][nf_resnet50][lp44]',
    '[t2][nf_resnet50][lp55]',
    '[t2][nf_resnet50][lp66]',
    '[t2][nf_resnet50][lp77]',
]

if '+' in dataset:  # UP-MPF
    loads = ['rot_pretrain_49-1-94.36',
             'rot_pretrain_49-2-95.93',
             'rot_pretrain_49-3-96.35',
             'rot_pretrain_49-4-96.47',
             'rot_pretrain_49-5-96.54',
             'rot_pretrain_49-6-96.93',
             ]
    temp = []
    for t in templates:
        for load in loads:
            temp.append(op.join(t, load))
    templates = temp
else:  # add PF templates
    textual_templates = [f'[t1][ps11{ph}]', '[t2]']
    templates.extend(textual_templates)


for t in templates:
    root = os.path.join(out_dir, dataset, train_dev + t)

    results = []
    for lr in lr_list:
        path = op.join(root, str(lr))
        file_list = os.listdir(path)
        png_files = [f for f in file_list if f[-9:] in ['.ckpt.png', '.ckpt.txt']]

        result = [p.search(s).groups() for s in png_files]
        result = [[float(i) for i in tup] for tup in result]
        result = torch.tensor(result)

        max_result, _ = torch.max(result, dim=0)
        result, _ = torch.sort(result, dim=0)
        result = result[1:-1, ]
        std = torch.std(result, dim=0)
        mean_result = torch.mean(result, dim=0)
        result = torch.cat([max_result, mean_result, std], dim=0)
        results.append(result)

    results = torch.stack(results, dim=0)
    final_result, order = torch.max(results, dim=0)
    assert final_result[0] == results[order[0]][0]
    final_result[6] = results[order[3]][6]
    final_result[7] = results[order[4]][7]
    final_result[8] = results[order[5]][8]

    max_res = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[0:3]])
    mean_res = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[3:6]])
    std = '_'.join(['{:.2f}'.format(i.item()) for i in final_result[6:9]])
    file_name = '__'.join([max_res, mean_res, std]) + '.txt'

    with open(op.join(root, file_name), 'w', encoding='utf-8') as f:
        f.write(str(order) + '\n')
        f.write('|            Max            |           Mean           |           Std            |\n')
        f.write('|   Acc  |  Mac-F1 | Wtd-F1 |  Acc  |  Mac-F1 | Wtd-F1 |  Acc  |  Mac-F1 | Wtd-F1 |\n')
        for line in results:
            f.write('|')
            for i in line:
                f.write('  {:5.2f}  '.format(i.item()))
            f.write('|\n')
