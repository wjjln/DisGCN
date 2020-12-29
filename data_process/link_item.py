import numpy as np
from collections import defaultdict
from itertools import product
import random
from time import time
import setproctitle
import argparse
parser = argparse.ArgumentParser(description="Link Items")
parser.add_argument('--data_name', type=str, default='Yelp')
args = parser.parse_args()
data_name = args.data_name
setproctitle.setproctitle('link-item-{}@linian'.format(data_name))

filename = 'data/{}/{}.train.rating'.format(data_name, data_name)
with open(filename) as f:
    data = [[int(x.split('\t')[0]), int(x.split('\t')[1])] for x in f.readlines()]
filename = 'data/{}/{}_data_size.txt'.format(data_name, data_name)
with open(filename) as f:
    num_users, num_items, _ = f.readline().split('\t')
    num_users, num_items = int(num_users), int(num_items)

item_inter_users = defaultdict(set)
for [u, i] in data:
    item_inter_users[i].add(u)

start = time()
i_link_i = []
for ind, [i1, i2] in enumerate(product(range(num_items), range(num_items))):
    if len(item_inter_users[i1]&item_inter_users[i2]) >= 8:
        i_link_i.append([i1, i2])
    if ind%10000000 == 0:
        print(ind, time()-start)
        start = time()

print('# of item links: {}'.format(len(i_link_i)))

filename = 'data/{}/{}_item.links'.format(data_name, data_name)
with open(filename, 'w') as f:
    for i1, i2 in i_link_i:
        f.writelines(str(i1)+'\t'+str(i2)+'\n')
print('write {} done'.format(filename))