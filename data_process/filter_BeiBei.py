import random
import numpy as np
from collections import defaultdict
from scipy.io import loadmat
from sklearn import preprocessing
from time import time
from sklearn.model_selection import train_test_split
num_inter = 5
file_path = '/data4/zhangjun/GBDataset/NewBeiBei/for_group_purchase/'
def read_data(file):
    with open(file_path+file) as f:
        data = []
        for x in f.readlines():
            tmp = [int(s) for s in x.split('\t')]
            data.append(tmp)
        return data

train, tune, test = read_data('train.txt'), read_data('tune.txt'), read_data('test.txt')
valid = train+tune+test
consumed_dict = defaultdict(set)
for v in valid:
    consumed_dict[v[0]].add(v[1])
    for u in v[2:]:
        consumed_dict[u].add(v[1])
records = [[u, i] for u in consumed_dict for i in consumed_dict[u]]
social = read_data('social_relation.txt')

for k in range(100):
    print('---------------------{}---------------------'.format(k))
    user_consumed_item, item_consumed_user = defaultdict(set), defaultdict(set)
    user_consumed_item_len, item_consumed_user_len = defaultdict(set), defaultdict(set)
    U, I = [], []
    for u, i in records:
        user_consumed_item[u].add(i)
        item_consumed_user[i].add(u)
    for u in user_consumed_item.keys():
        tmp = len(user_consumed_item[u])
        user_consumed_item_len[u] = tmp
        U.append(tmp)
    for i in item_consumed_user.keys():
        tmp = len(item_consumed_user[i])
        item_consumed_user_len[i] = tmp
        I.append(tmp)
    if (np.array(U) >= num_inter).all() and (np.array(I) >= num_inter).all():
        print('done')
        break

    records_5 = []
    for u, i in records:
        if user_consumed_item_len[u] >= num_inter and item_consumed_user_len[i] >= num_inter:
            records_5.append([u, i])
    records = records_5

    print('# of interactions: {}'.format(len(records)))

user_le = preprocessing.LabelEncoder()
item_le = preprocessing.LabelEncoder()
records = np.array(records)
user_le.fit(records[:, 0])
item_le.fit(records[:, 1])
reserved_user = user_le.classes_
reserved_item = item_le.classes_
print(f'#users: {len(reserved_user)}, #items: {len(reserved_item)}')
records[:, 0] = user_le.transform(records[:, 0])
records[:, 1] = item_le.transform(records[:, 1])

links = []
for u, f in social:
    if u in reserved_user and f in reserved_user:
        links.append([u, f])
links = np.array(links)
links[:, 0] = user_le.transform(links[:, 0])
links[:, 1] = user_le.transform(links[:, 1])

user_consumed_item = defaultdict(set)
for u, i in records:
    user_consumed_item[u].add(i)
train, test, val = [], [], []
for u in user_consumed_item.keys():
    samples = user_consumed_item[u]
    if len(samples) < 10:
        t = random.sample(samples, 1)
        val.append([u, t[0]])
        left = samples - set(t)
        t = random.sample(left, 2)
        test.append([u, t[0]])
        test.append([u, t[1]])
        left = left - set(t)
        for i in left:
            train.append([u, i])
    else:
        samples = list(samples)
        tt, t = train_test_split(samples, test_size=0.3, random_state=123)
        for tmp in tt:
            train.append([u, tmp])
        tt, t = train_test_split(t, test_size=1.0/3, random_state=124)
        for tmp in tt:
            test.append([u, tmp])
        for tmp in t:
            val.append([u, tmp])
print(len(train), len(val), len(test))

train_dict = defaultdict(set)
for u, i in train:
    train_dict[u].add(i)
from itertools import combinations
ufi = []
for v in valid:
    if v[1] not in reserved_item:
        continue
    u = list(filter(lambda s: s in reserved_user, [v[0]] + v[2:]))
    if len(u) >= 2:
        for user, friend in combinations(u, 2):
            ufi.append((user, friend, v[1]))
ufi = np.array(ufi)
ufi[:, 0] = user_le.transform(ufi[:, 0])
ufi[:, 1] = user_le.transform(ufi[:, 1])
ufi[:, 2] = item_le.transform(ufi[:, 2])
final_ufi = []
for u, f, i in ufi:
    if (i in train_dict[u]) and (i in train_dict[f]):
        final_ufi.append([u, f, i])
print(len(final_ufi))

def write_data(file, links):
    with open(file, 'w') as f:
        for x in links:
            write_str = ''
            for xx in x[:len(x)-1]:
                write_str += f'{str(xx)}\t'
            write_str += f'{x[-1]}\n'
            f.write(write_str)

write_data('data/BeiBei/BeiBei.train.rating', train)
write_data('data/BeiBei/BeiBei.test.rating', test)
write_data('data/BeiBei/BeiBei.val.rating', val)
write_data('data/BeiBei/BeiBei_all.links', links)
write_data('data/BeiBei/BeiBei_ufi.links', final_ufi)
print('write done')