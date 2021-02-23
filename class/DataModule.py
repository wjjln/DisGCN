
from collections import defaultdict
import numpy as np
from time import time
import random
import math
import operator
import pandas as pd
import sys
sys.path.append('/data4/linian/Social_Rec/data/')

class DataModule():
    def __init__(self, conf, filename, links_filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.links_filename = links_filename
        self.index = 0
        self.index_link = 0
        self.print_once = True
        self.loss_turn = 0

###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self):
        data_dict = {}
        model_name = self.conf.model_name
        need_S_idx = (model_name in ['gcncsr', 'diffnet', 'graphrec', 'socialbpr', 'samn', 'sorec', 'disengcn', 'disbpr', 'disgcn'])
        need_S_values = (model_name in ['gcncsr', 'diffnet', 'socialbpr', 'disengcn', 'disbpr', 'disgcn'])
        need_C_idx = (model_name in ['gcncsr', 'diffnet', 'graphrec', 'ngcf', 'lightgcn', 'gcn', 'gcmc', 'disengcn', 'disbpr', 'disgcn'])
        need_C_values = (model_name in ['gcncsr', 'diffnet', 'ngcf', 'lightgcn', 'gcn', 'gcmc', 'disengcn', 'disbpr', 'disgcn'])
        if need_S_idx:
            self.generateSocialNeighborsSparseMatrix(need_S_values)
            data_dict['social_edges_user0'] = self.social_edges_user0
            data_dict['social_edges_user1'] = self.social_edges_user1
            if need_S_values:
                data_dict['social_neighbors_values_input'] = self.social_neighbors_values_input
        if need_C_idx:
            self.generateConsumedItemsSparseMatrix(need_C_values)
            data_dict['interaction_user_indices'] = self.interaction_user_indices
            data_dict['interaction_item_indices'] = self.interaction_item_indices
            if need_C_values:
                data_dict['consumed_items_values_input'] = self.consumed_items_values_input
                data_dict['inter_users_values_input'] = self.inter_users_values_input
        if model_name in ['lightgcn', 'gcn', 'ngcf']:
            data_dict['user_self_value'] = self.user_self_value
            data_dict['item_self_value'] = self.item_self_value
        if  model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            data_dict['social_edges_user_neg'] = self.negative_link
        if model_name in ['gcmc', 'ngcf'] and self.conf.user_side:
            data_dict['user_side_information'] = self.u_emb

        if model_name in ['disgcn', 'csr']:
            data_dict['u_input'] = self.u_input
            data_dict['f_input'] = self.f_input
            data_dict['i_input'] = self.i_input
            data_dict['seg_in_friends'] = self.seg_in_friends
            data_dict['ufi_att'] = self.ufi_att
            data_dict['int_att'] = self.int_att
            data_dict['social_att_idx'] = self.social_att_idx
            data_dict['social_att'] = self.social_att
            
        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initializeRankingVT(self, d_train):
        self.readData()
        self.arrangePositiveData()
        # self.generateVTNegative(d_train)

    def initalizeRankingEva(self):
        self.readData()
        self.arrangePositiveData()

    def construct_ground_truth(self):
        ground_truth = np.zeros([len(self.batch_user_list), self.conf.num_items])
        for uid, u in enumerate(self.batch_user_list):
            ground_truth[uid, list(self.user_consumed_items[u])] = 1
        return ground_truth

    def linkedMap(self):
        if self.loss_turn == 0:
            self.data_dict['USER_LIST'] = self.user_list
            self.data_dict['ITEM_LIST'] = self.item_list
            self.data_dict['ITEM_NEG_INPUT'] = self.item_neg_list
            if self.conf.model_name in ['Geobpr']:
                self.data_dict['Geo0_list'] = self.Geo0_list
                self.data_dict['Geo1_list'] = self.Geo1_list
                self.data_dict['cor_idx'] = self.cor_idx
            if self.conf.model_name in ['disengcn', 'disbpr', 'disgcn'] and 'train' in self.filename:
                self.data_dict['cor_idx'] = self.cor_idx
            if not self.terminal_flag and self.conf.model_name in ['disbpr', 'disgcn'] and self.conf.social_loss:
                self.loss_turn = 1
        else:
            self.data_dict['ufi_u'] = self.ufi_u
            self.data_dict['ufi_f'] = self.ufi_f
            self.data_dict['ufi_i'] = self.ufi_i
            self.data_dict['ufi_j'] = self.ufi_j
            if self.conf.g:
                self.data_dict['ufi_g'] = self.ufi_g
            if not self.terminal_flag:
                self.loss_turn = 0
        if self.conf.model_name in ['gcncsr', 'esgcn', 'sbpr', 'sorec'] and self.conf.social_loss:
            self.data_dict['USER_SOCIAL_LOSS_IDX'] = self.user_social_loss_idx
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_USER_LIST'] = self.eva_user_list
        self.data_dict['EVA_ITEM_LIST'] = self.eva_item_list

    def readData(self):
        self.total_data = []
        with open(self.filename) as f:
            for x in f.readlines():
                u, i = x.split('\t')[:2]
                u, i = int(u), int(i)
                self.total_data.append([u, i])
        self.total_link = []
        with open(self.links_filename) as f:
            for x in f.readlines():
                tmp = x.split('\t')[:2]
                u, f = int(tmp[0]), int(tmp[1])
                self.total_link.append([u, f])
        if self.conf.model_name in ['disbpr', 'disgcn', 'csr'] and 'train' in self.filename:
            ufi = []
            with open('data/{0}/{0}_ufi.links'.format(self.conf.data_name)) as file:
                for x in file.readlines():
                    tmp = x.split('\t')
                    u, f, i = int(tmp[0]), int(tmp[1]), int(tmp[2])
                    ufi.append([u, f, i])
            ufi = np.array(ufi)
            ufi = np.unique(np.concatenate([np.concatenate([ufi[:, :2], np.fliplr(ufi[:, :2])], 0), np.expand_dims(np.tile(ufi[:, 2], 2), 1)], 1), axis=0)
            self.ufi = ufi[(ufi[:, 0]*self.conf.num_users+ufi[:, 1]).argsort()]
        self.total_user_list = list(range(0, self.conf.num_users))
        self.total_item_list = list(range(0, self.conf.num_items))
        if self.conf.model_name in ['Geobpr']:
            with open('/home/linian/KGAT/Data/MDB_yelp/item_coor.txt') as f:
                item_coor = pd.read_csv(f, sep=' ')
            self.coor = np.array(item_coor[['latitude', 'longitude']])
        if self.conf.model_name in ['gcmc', 'ngcf'] and self.conf.user_side:
            self.u_emb = np.zeros([self.conf.num_users, self.conf.dimension])
            emb_name_f = 'data/{}/deepwalk'.format(self.conf.data_name)
            emb_name = emb_name_f + '.txt' if self.conf.dimension == 64 else emb_name_f + '_{}.txt'.format(self.conf.dimension)
            with open(emb_name) as f:
                for x in f.readlines()[1:]:
                    tmp = np.fromstring(x, sep=' ')
                    self.u_emb[int(tmp[0])] = tmp[1:]

    def arrangePositiveData(self):
        self.positive_data = np.array(self.total_data)
        self.positive_data = self.positive_data[(self.positive_data[:,0]*self.conf.num_items+self.positive_data[:, 1]).argsort()]
        user_consumed_items = defaultdict(set)
        item_inter_users = defaultdict(set)
        for [u, i] in self.positive_data:
            user_consumed_items[u].add(i)
            item_inter_users[i].add(u)
        self.user_consumed_items = user_consumed_items
        self.item_inter_users = item_inter_users
        positive_link = np.array(self.total_link)
        positive_link = np.unique(np.concatenate([positive_link, np.fliplr(positive_link)], 0), axis=0)
        self.positive_link = positive_link[(positive_link[:,0]*self.conf.num_users+positive_link[:,1]).argsort()]
        print('# of interactions: {}, # of friendship: {}'.format(len(self.positive_data), len(self.positive_link)))
        user_friends = defaultdict(set)
        for [u, fr] in self.positive_link:
            user_friends[u].add(fr)
        self.user_friends = user_friends
        if self.conf.model_name in ['gcncsr', 'sbpr'] and self.conf.social_loss:
            L = len(self.positive_link)
            num_split = 100
            step = L//num_split+1
            self.split_idx = [0]
            for i in range(num_split):
                end = self.split_idx[-1] + step
                self.split_idx.append(np.min([L, end]))

        if self.conf.model_name in ['disgcn', 'csr'] and 'train' in self.filename:
            from sklearn.preprocessing import LabelEncoder
            ufi = self.ufi
            label = ufi[:, 0]*self.conf.num_users+ufi[:, 1]
            ufi_le = LabelEncoder()
            self.seg_in_friends = ufi_le.fit_transform(label)
            self.u_input, self.f_input, self.i_input = ufi[:, 0], ufi[:, 1], ufi[:, 2]
            _, ufi_att = np.unique(ufi[:, :2], axis=0, return_counts=True)
            self.ufi_att = np.repeat(1.0/ufi_att, ufi_att).reshape([-1, 1])

            _, int_att = np.unique(self.positive_link[:, 0], axis=0, return_counts=True)
            self.int_att = np.repeat(1.0/int_att, int_att).reshape([-1, 1])
            self.social_att_idx = np.unique(ufi[:, :2], axis=0)
            _, social_att = np.unique(self.social_att_idx[:, 0], axis=0, return_counts=True)
            self.social_att = np.repeat(1.0/social_att, social_att).reshape([-1, 1])

    def sample_neg(self, num_neg, num_all, cond_sets):
        tmp = []
        for _ in range(num_neg):
            j = np.random.randint(num_all)
            while True:
                if j not in cond_sets:
                    break
                j = np.random.randint(num_all)
            tmp.append([j])
        return tmp

    def generateTrainNegative(self):
        num_items = self.conf.num_items
        num_users = self.conf.num_users
        num_negatives = self.conf.num_negatives
        negative_data = []
        for u, _ in self.positive_data:
            tmp = self.sample_neg(num_negatives, num_items, self.user_consumed_items[u])
            negative_data.append(tmp)
        self.negative_data = np.array(negative_data)
        if self.conf.model_name in ['disbpr', 'disgcn'] and self.conf.social_loss:
            negative_ufi = []
            for u, f, _ in self.ufi:
                tmp = self.sample_neg(num_negatives, num_items, self.user_consumed_items[u].union(self.user_consumed_items[f]))
                negative_ufi.append(tmp)
            self.negative_ufi_i = np.array(negative_ufi)
            if self.conf.g:
                negative_ufi = []
                for u, _, i in self.ufi:
                    tmp = self.sample_neg(num_negatives, num_users, self.user_friends[u].union(self.item_inter_users[i]))
                    negative_ufi.append(tmp)
                self.negative_ufi_f = np.array(negative_ufi)
        if self.conf.model_name in ['Geobpr']:
            Geo_item0, Geo_item1 = [], []
            for i in self.positive_data[:, 1]:
                x = np.random.randint(0, num_items, 10)
                x_dis = np.sum(np.square(self.coor[x] - self.coor[i]), 1)
                Geo_item0.append(np.argmin(x_dis))
                Geo_item1.append(np.argmax(x_dis))
            self.Geo_item0 = np.reshape(np.array(Geo_item0), [-1, 1])
            self.Geo_item1 = np.reshape(np.array(Geo_item1), [-1, 1])
        if self.conf.model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            negative_link = []
            for u, _ in self.positive_link:
                tmp = self.sample_neg(num_negatives, num_users, [self.user_friends[u]])
                negative_link.append(tmp)
            self.negative_link = np.array(negative_link)
    
    def generateVTNegative(self, d_train):
        num_items = self.conf.num_items
        num_users = self.conf.num_users
        num_negatives = self.conf.num_negatives
        negative_data = []
        for u, _ in self.positive_data:
            tmp = self.sample_neg(num_negatives, num_items, [self.user_consumed_items[u], d_train.user_consumed_items[u]])
            negative_data.append(tmp)
        self.negative_data = np.array(negative_data)
        if self.conf.model_name in ['Geobpr']:
            Geo_item0, Geo_item1 = [], []
            for i in self.positive_data[:, 1]:
                x = np.random.randint(0, num_items, 10)
                x_dis = np.sum(np.square(self.coor[x] - self.coor[i]), 1)
                Geo_item0.append(np.argmin(x_dis))
                Geo_item1.append(np.argmax(x_dis))
            self.Geo_item0 = np.reshape(np.array(Geo_item0), [-1, 1])
            self.Geo_item1 = np.reshape(np.array(Geo_item1), [-1, 1])
        if self.conf.model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            negative_link = []
            for u, _ in self.positive_link:
                tmp = self.sample_neg(num_negatives, num_users, [self.user_friends[u]])
                negative_link.append(tmp)
            self.negative_link = np.array(negative_link)
    
    def getTrainRankingBatch(self):
        positive_data = self.positive_data
        len_positive_data = len(positive_data)
        batch_size = self.conf.training_batch_size
        index = self.index

        if self.loss_turn == 0:
            tmp = min((len_positive_data, index+batch_size))
            batch_data = positive_data[index:tmp]
            self.item_neg_list = self.negative_data[index:tmp]
            if self.conf.model_name in ['Geobpr']:
                    self.Geo0_list = self.Geo_item0[index:tmp]
                    self.Geo1_list = self.Geo_item1[index:tmp]
            if self.conf.model_name in ['disgcn', 'disbpr', 'disengcn'] and 'train' in self.filename:
                cor_user = np.array(random.sample(self.total_user_list, tmp-index))
                cor_item = np.array(random.sample(self.total_item_list, tmp-index)) + self.conf.num_users
                self.cor_idx = np.concatenate([cor_user, cor_item], 0)
            if tmp >= len_positive_data:
                self.index = 0
                self.terminal_flag = 0
            else:
                self.index = index + batch_size
            
            self.user_list = batch_data[:, 0]
            self.item_list = batch_data[:, 1]

        else:
            ufi = self.ufi
            len_ufi = len(self.ufi)
            ufi_batch_size = int(np.ceil(len_ufi/(np.ceil(len_positive_data/batch_size))))
            if self.print_once:
                print(f'ufi_batch_size: {ufi_batch_size}')
                self.print_once = False
            index_link = self.index_link
            tmp = min((len_ufi, index_link+ufi_batch_size))
            batch_ufi = ufi[index_link:tmp]
            self.ufi_j = self.negative_ufi_i[index_link:tmp]
            if self.conf.g:
                self.ufi_g = self.negative_ufi_f[index_link:tmp]
            self.ufi_u = batch_ufi[:, 0]
            self.ufi_f = batch_ufi[:, 1]
            self.ufi_i = batch_ufi[:, 2]
            if tmp >= len_ufi:
                self.index_link = 0
                self.terminal_flag = 0
            else:
                self.index_link = index_link + ufi_batch_size

        if self.conf.model_name in ['gcncsr', 'sorec'] and self.conf.social_loss:
            positive_link = self.positive_link
            len_positive_link = len(positive_link)
            link_batch_size = int(np.ceil(len_positive_link/(np.ceil(len_positive_data/batch_size))))
            # link_batch_size = 2048
            index_link = self.index_link
            if index_link + link_batch_size < len_positive_link:
                batch_link = range(index_link, index_link+link_batch_size)
                self.index_link = index_link + link_batch_size
            else:
                batch_link = range(index_link, len_positive_link)
                self.index_link = 0
            self.user_social_loss_idx = batch_link
        
    def getEvaBatch(self):
        batch_user_list = []
        len_batch = 0
        index = self.index
        batch_size = self.conf.evaluate_batch_size
        if index + batch_size < self.conf.num_users:
            batch_user_list = self.total_user_list[index:index+batch_size]
            self.index = index + batch_size
            len_batch = batch_size
        else:
            self.terminal_flag = 0
            batch_user_list = self.total_user_list[index:self.conf.num_users]
            len_batch = self.conf.num_users - index
            self.index = 0
        self.batch_user_list = np.array(batch_user_list)
        self.len_batch = len_batch
        num_items = self.conf.num_items
        if self.conf.model_name in ['gmf', 'mlp', 'neumf', 'hbpr', 'graphrec', 'disengcn']:
            self.eva_user_list = np.repeat(batch_user_list, num_items)
            self.eva_item_list = np.tile(range(num_items), len_batch)
        else:
            self.eva_user_list = self.batch_user_list
            self.eva_item_list = None


    def generateSocialNeighborsSparseMatrix(self, values=True):
        model_name = self.conf.model_name
        self.social_edges_user0 = np.reshape(self.positive_link[:, 0], [-1, 1])
        self.social_edges_user1 = np.reshape(self.positive_link[:, 1], [-1, 1])
        if values:
            social_neighbors_values_input = []
            if model_name in ['gcncsr', 'disengcn']:
                for uid, u0 in enumerate(self.social_edges_user0):
                    u0 = int(u0)
                    u1 = int(self.social_edges_user1[uid])
                    social_neighbors_values_input.append(1.0/math.sqrt((len(self.user_friends[u0])+1)*(len(self.user_friends[u1])+1)))
                    # social_neighbors_values_input.append(1.0/len(self.user_friends[u0]))
            if model_name in ['diffnet', 'socialbpr', 'disbpr', 'disgcn']:
                for u in self.social_edges_user0:
                    u = int(u)
                    social_neighbors_values_input.append(1.0/len(self.user_friends[u]))
            self.social_neighbors_values_input = np.array(social_neighbors_values_input).astype(np.float32)



    def generateConsumedItemsSparseMatrix(self, values=True):
        model_name = self.conf.model_name
        self.interaction_user_indices = np.expand_dims(self.positive_data[:, 0], 1)
        self.interaction_item_indices = np.expand_dims(self.positive_data[:, 1], 1)
        if values:
            consumed_items_values_input = []
            inter_users_values_input = []
            if model_name in ['diffnet', 'graphrec', 'gcncsr', 'disbpr', 'disgcn']:
                for id, u in enumerate(self.interaction_user_indices):
                    u = int(u)
                    i = int(self.interaction_item_indices[id])
                    consumed_items_values_input.append(1.0/len(self.user_consumed_items[u]))
                    inter_users_values_input.append(1.0/len(self.item_inter_users[i]))
            if model_name in ['ngcf', 'gcn', 'lightgcn', 'gcmc', 'disengcn']:
                self.user_self_value = np.zeros([self.conf.num_users, 1]).astype(np.float32)
                self.item_self_value = np.zeros([self.conf.num_items, 1]).astype(np.float32)
                for u, i in self.positive_data:
                    u, i = int(u), int(i)
                    num_i = len(self.user_consumed_items[u])+1
                    num_u = len(self.item_inter_users[i])+1
                    L = 1.0/math.sqrt(num_i*num_u)
                    consumed_items_values_input.append(L)
                    inter_users_values_input.append(L)
                    self.user_self_value[u, 0] = 1.0/num_i
                    self.item_self_value[i, 0] = 1.0/num_u
            self.consumed_items_values_input = np.array(consumed_items_values_input).astype(np.float32)
            self.inter_users_values_input = np.array(inter_users_values_input).astype(np.float32)