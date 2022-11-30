'''
Select sentiment labels for images and texts by majority voting first
And then remove examples which have inconsistent sentiment between image and text
'''

# Please excute following commands first
# $ unzip MVSA-multiple.zip
# $ mv data MVSA-M_data

fin = open('labelResultAll.txt', 'r', encoding='utf-8')
fout = open('MVSA-M_id_label.txt', 'w' , encoding='utf-8')

fout.write('id	label\n')
lines = fin.readlines()
lines.pop(0)
count0 = 0
count1 = 0
for l in lines:
    id, senti1, senti2, senti3 = l.split()
    t1, i1 = senti1.split(',')
    t2, i2 = senti2.split(',')
    t3, i3 = senti3.split(',')

    def f(a, b, c):
        d = {'negative': 0, 'neutral': 0, 'positive': 0}
        d[a] += 1
        d[b] += 1
        d[c] += 1
        for i in d.keys():
            if d[i] == 2 or d[i] == 3:
                return i
        # print(a, b, c)
        return None

    t_senti = f(t1, t2, t3)
    i_senti = f(i1, i2, i3)
    if t_senti is None or i_senti is None:
        count0 += 1
        continue

    if (t_senti == 'positive' and i_senti == 'negative') or (i_senti == 'positive' and t_senti == 'negative'):
        count1 += 1  # conflict
    elif t_senti == i_senti:
        fout.write(f'{id}\t{t_senti}\n')
    elif t_senti == 'neutral':
        fout.write(f'{id}\t{i_senti}\n')
    elif i_senti == 'neutral':
        fout.write(f'{id}\t{t_senti}\n')
    else:
        raise RuntimeError('Error')

print('[#] All three can\'t agree:', count0)
print('[#] Conflict between text and image:', count1)

fin.close()
fout.close()
