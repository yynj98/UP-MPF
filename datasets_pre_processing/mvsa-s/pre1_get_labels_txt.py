'''
Remove examples which have inconsistent sentiment between image and text
'''

# Please excute following commands first
# $ unzip MVSA-Single.zip
# $ mv data MVSA-S_data

fin = open('labelResultAll.txt', 'r', encoding='utf-8')
fout = open('MVSA-S_id_label.txt', 'w' , encoding='utf-8')

fout.write('id	label\n')
lines = fin.readlines()
lines.pop(0)
count = 0
for l in lines:
    id, senti = l.split()
    s1, s2 = senti.split(',')
    if (s1 == 'positive' and s2 == 'negative') or (s2 == 'positive' and s1 == 'negative'):
        count += 1
    elif s1 == s2:
        fout.write(f'{id}\t{s1}\n')
    elif s1 == 'neutral':
        fout.write(f'{id}\t{s2}\n')
    elif s2 == 'neutral':
        fout.write(f'{id}\t{s1}\n')
    else:
        raise RuntimeError('Error')
print(count)

fin.close()
fout.close()
