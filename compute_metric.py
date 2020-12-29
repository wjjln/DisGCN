from metric import *
from collections import defaultdict
import torch
import os

def compute_metric(scores, groud_truth, conf):
    topk = conf.topk
    metric_dict = defaultdict(list)
    for k in topk:
        metric_dict['Recall'].append(Recall(k))
        metric_dict['NDCG'].append(NDCG(k))
        metric_dict['MRR'].append(MRR(k))
    metric_name = {'Recall':[], 'NDCG':[], 'MRR':[]}

    with torch.no_grad():
        for i in range(len(topk)):
            for name in metric_name.keys():
                m = metric_dict[name][i]
                m.start()
                m(scores, groud_truth)
                # print '{}:{}'.format(m.get_title(), m.metric),
                m.stop()
                metric_name[name].append(m.metric)
                # if conf.test:
                #     dirs = 'RQ3/{}/{}/'.format(conf.data_name, conf.model_name)
                #     if not os.path.exists(dirs):
                #         os.makedirs(dirs)
                #     filename = dirs+'{}_{}.npy'.format(name, topk[i])
                #     T = m.each.cpu().numpy()
                #     if os.path.isfile(filename):
                #         tmp = np.load(filename)
                #         T = np.concatenate([tmp, T], 0)
                #     np.save(filename, T)
    return metric_name