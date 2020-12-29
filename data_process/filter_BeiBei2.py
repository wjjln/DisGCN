import random
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from time import time
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
from copy import copy

num_inter = 10
data_name = 'BeiBei2'
path = f'data/{data_name}/'
# records = pd.concat([pd.read_csv(path+file) for file in os.listdir(path) if 'non' in file and 'add' not in file])
share = pd.concat([pd.read_csv(path+file) for file in os.listdir(path) if 'ref' in file and 'non' not in file and 'add' not in file])

# records_add = pd.read_csv(path+'nonref_add.csv')
# records_add['info'] = records_add['info'].apply(lambda x: eval(x))
# records_add = records_add.explode('info')
# records_add.rename(columns={'info':'iid'}, inplace=True)
# records = pd.concat([records, records_add])
# records = records.dropna(axis=0,how='any')
# records = records.astype('int32')

share_add = pd.read_csv(path+'ref_add.csv')
share_add['info'] = share_add['info'].apply(lambda x: eval(x))
share_add = share_add.explode('info')
share_add[['ruid', 'iid', 'indicators']] = pd.DataFrame(share_add['info'].tolist(), index=share_add.index)   
share_add = share_add.drop(columns=['info'])
share_add = share_add[share_add['ruid'].str.contains('^[0-9]+$', regex=True)]
share = pd.concat([share, share_add])
share = share.dropna(axis=0,how='any')
share = share.astype('int32')

# share_buy = share[share['indicators']==1]
# buy_u = np.unique(np.array(share_buy[['uid']], dtype=np.int32))
# buy_f = np.unique(np.array(share_buy[['ruid']], dtype=np.int32))
# buy_uf = np.unique(np.concatenate([buy_u, buy_f], 0))
# buy_i = np.unique(np.array(share_buy[['iid']], dtype=np.int32))

# def filter_fun(row):
#     if row['uid'] in buy_uf and row['iid'] in buy_i:
#         return True
#     else:
#         return False
# pandarallel.initialize(nb_workers=16)
# records = records[records.parallel_apply(filter_fun, axis=1)]

# def df_unique(df):
#     s = set()
#     for d in df:
#         s = s.union(set(np.unique(np.array(df, dtype=np.int32)).tolist()))
#     return s
# tmp_records, tmp_share = copy(records), copy(share)
# pandarallel.initialize(nb_workers=16)
# N = 2
# for k in range(N):
#     records_u, share_u = df_unique([records['uid']]), df_unique([share['uid'], share['ruid']])
#     records_i, share_i = df_unique([records['iid']]), df_unique([share['iid']])
#     U = records_u.intersection(share_u)
#     I = records_i.intersection(share_i)
#     print(len(U), len(I))
#     records = records[records.parallel_apply(lambda row: True if row['uid'] in U and row['iid'] in I else False, axis=1)]
#     share = share[share.parallel_apply(lambda row: True if row['uid'] in U and row['ruid'] in U and row['iid'] in I else False, axis=1)]
#     print(len(records), len(share))
#     if k < N-1:
#         share = share.sample(frac=0.5, random_state=123)
#         records = records.sample(frac=0.5, random_state=124)

# records = np.array(records[['uid', 'iid']], dtype=np.int32)
share = np.unique(np.array(share[['ruid', 'uid', 'iid']], dtype=np.int32), axis=0)
records = np.unique(np.concatenate([share[:, 1:], share[:, [0, 2]]], 0), axis=0)
social = np.unique(share[:, :2], axis=0)
print(f'records: {len(records)}, share: {len(share)}, social: {len(social)}')

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
import pickle
pickle.dump(user_le, open('user_le.pkl', 'wb'))
pickle.dump(item_le, open('item_le.pkl', 'wb'))
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
ufi = [v for v in share if v[0] in reserved_user and v[1] in reserved_user and v[2] in reserved_item]
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

write_data(f'data/{data_name}/{data_name}.train.rating', train)
write_data(f'data/{data_name}/{data_name}.test.rating', test)
write_data(f'data/{data_name}/{data_name}.val.rating', val)
write_data(f'data/{data_name}/{data_name}_all.links', links)
write_data(f'data/{data_name}/{data_name}_ufi.links', final_ufi)
print('write done')