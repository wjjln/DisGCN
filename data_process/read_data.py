import numpy as np
from collections import defaultdict


def read_data(data_name, plus=False):
    work_path = f'/data4/linian/Social_Rec/data/{data_name}/{data_name}'
    user_consumed_train = defaultdict(set)
    user_consumed_val = defaultdict(set)
    user_consumed_test = defaultdict(set)
    user_friend = defaultdict(set)
    item_inter = defaultdict(set)

    with open(work_path+'.train.rating') as f:
        for x in f.readlines():
            u, i = x.split('\t')[:2]
            u, i = int(u), int(i)
            if plus:
                u += 1
                i += 1
            user_consumed_train[u].add(i)
            item_inter[i].add(u)
    
    with open(work_path+'.val.rating') as f:
        for x in f.readlines():
            u, i = x.split('\t')[:2]
            u, i = int(u), int(i)
            if plus:
                u += 1
                i += 1
            user_consumed_val[u].add(i)
    
    with open(work_path+'.test.rating') as f:
        for x in f.readlines():
            u, i = x.split('\t')[:2]
            u, i = int(u), int(i)
            if plus:
                u += 1
                i += 1
            user_consumed_test[u].add(i)

    with open(work_path+'_all.links') as f:
        for x in f.readlines():
            u, f = x.split('\t')[:2]
            u, f = int(u), int(f)
            if plus:
                u += 1
                f += 1
            user_friend[u].add(f)
            user_friend[f].add(u)
    return user_consumed_train, user_consumed_val, user_consumed_test, item_inter, user_friend