from collections import defaultdict
from visdom import Visdom
import numpy as np

model_name = 'disbpr'
test_name = 'fs-l2sum-g'
data_name = 'BeiBei2'
log_path = '{}_log/{}/dim32_{}.log'.format(data_name, model_name, test_name)
data = defaultdict(list)
model = dict()
reg, lr = 0, 0
read_metric = 1
main_method = (model_name in ['gcncsr', 'disbpr', 'disgcn'])
with open(log_path) as f:
    for s in f.readlines():
        x = s.replace(' ', '').strip('\n')
        if 'reg:' in x:
            model[(reg, lr)] = data
            data = defaultdict(list)
            reg, lr = x.split(',')
            reg = float(reg[4:])
            lr = lr.strip('-')
            lr = float(lr[3:])
        elif 'Epoch:' in x:
            if main_method:
                _, _, train_loss, social_loss = x.split(',')
            else:
                _, _, train_loss = x.split(',')
            train_loss = float(train_loss[10:])
            data['train_loss'].append(train_loss)
            read_metric = 1
            if main_method:
                data['social_loss'].append(float(social_loss[12:]))
        elif 'Recall@' in x and read_metric:
            Recall, NDCG, MRR = x.split(',')
            Recall, NDCG, MRR = float(Recall[10:]), float(NDCG[8:]), float(MRR[7:])
            data['Recall'].append(Recall)
            data['NDCG'].append(NDCG)
            data['MRR'].append(MRR)
        elif 'testmetric' in x:
            read_metric = 0
model[(reg, lr)] = data
print('get data done')

for k in model.keys():
    if k[0] > 0:
        vis = Visdom(port=1469, env='{}_reg{}lr{}-{}'.format(model_name, k[0], k[1], test_name))
        data = model[k]
        train_loss, val_loss, test_loss = data['train_loss'], data['val_loss'], data['test_loss']
        Recall, NDCG, MRR = np.array(data['Recall']), np.array(data['NDCG']), np.array(data['MRR'])
        X = np.arange(len(train_loss))+1
        vis.line(train_loss, X, win='train loss', opts={'title':'train loss'})
        if main_method:
            vis.line(data['social_loss'], X, win='social loss', opts={'title':'social loss'})
        n = len(Recall)/4
        X = np.linspace(5, 5*n, n)
        idx = np.linspace(1, 4*(n-1)+1, n).astype(np.int)
        vis.line(NDCG[idx], X, win='NDCG@20', opts={'title':'NDCG@20'})
        vis.line(Recall[idx], X, win='Recall@20', opts={'title':'Recall@20'})
        vis.line(MRR[idx], X, win='MRR@20', opts={'title':'MRR@20'})
        print('reg: {}, lr: {} to visdom done'.format(k[0], k[1]))