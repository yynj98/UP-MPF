'''
Filter https://, http://, @username, #hashtag
'''

import os
import re

data_dir = 'MVSA-S_data'
label_file = 'MVSA-S_id_label.txt'
fout_name = 'all.tsv'
stopwords = [',', '.', ';', '(', ')', 'RT']

fin = open(label_file, 'r', encoding='utf-8')
fout = open(fout_name, 'w' , encoding='utf-8')
fout.write('index	#1 Label	#2 ImageID	#3 String	#3 String\n')

filtered_by_len = []
conts = []
lines = fin.readlines()
lines.pop(0)
for line in lines:
    idx, label = line.split()
    txt = os.path.join(data_dir, idx + '.txt')
    # select utf-8 encoded text only, some texts are encoded in cp866
    with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
        l = f.readlines()
        assert len(l) == 1
        cont = l[0]
        
        cont = re.sub(r'https://\S+', '', cont)
        cont = re.sub(r'http://\S+', '', cont)
        cont = re.sub(r'\S*@\S+', '', cont)  # @username
        cont = re.sub(r'#\S+', '', cont)     # #hashtag
        cont = cont.split()
        cont = [w for w in cont if w not in stopwords]

        if len(cont) < 3:
            filtered_by_len.append(idx)
            continue
        cont = ' '.join(cont)
        conts.append(f'{idx}\t{label}\t{idx}.jpg\t{cont}\t\n')

conts.sort(key=lambda x: x.split('\t')[1])
for line in conts:
    fout.write(line)

fin.close()
fout.close()

# print(filtered_by_len)
print(len(filtered_by_len))
