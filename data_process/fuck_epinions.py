import random
import numpy as np
from collections import defaultdict
from scipy.io import loadmat
from sklearn import preprocessing
from time import time
from sklearn.model_selection import train_test_split
num_inter = 10
trust = loadmat('data/epinions/trustnetwork.mat')
social = trust['trustnetwork']
rating = loadmat('data/epinions/rating.mat')
check_ins = rating['rating']
records = check_ins[:, :2]
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
records[:, 0] = user_le.transform(records[:, 0])
records[:, 1] = item_le.transform(records[:, 1])
links = []
for u, f in social:
    if u in reserved_user and f in reserved_user:
        links.append([u, f])
links = np.array(links)
links[:, 0] = user_le.transform(links[:, 0])
links[:, 1] = user_le.transform(links[:, 1])
print(len(user_le.classes_), len(item_le.classes_))

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

with open('data/Epinions/Epinions_all.links', 'w') as f_link:
    for u, f in links:
        f_link.write(str(u)+'\t'+str(f)+'\n')

with open('data/Epinions/Epinions.train.rating', 'w') as f_train:
    for u, i in train:
        f_train.write(str(u)+'\t'+str(i)+'\n')
with open('data/Epinions/Epinions.test.rating', 'w') as f_test:
    for u, i in test:
        f_test.write(str(u)+'\t'+str(i)+'\n')
with open('data/Epinions/Epinions.val.rating', 'w') as f_val:
    for u, i in val:
        f_val.write(str(u)+'\t'+str(i)+'\n')
print('write done')