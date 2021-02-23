import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from base_model import base_model

class disgcn(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.social_neighbors_dense_shape = np.array([self.num_users, self.num_users]).astype(np.int64)
        interaction_user_indices = data_dict['interaction_user_indices']
        interaction_item_indices = data_dict['interaction_item_indices']
        self.consumed_items_indices_input = np.concatenate([interaction_user_indices, interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.consumed_items_dense_shape = np.array([self.num_users, self.num_items]).astype(np.int64)
        # self.inter_users_indices_input = np.fliplr(self.consumed_items_indices_input)
        # self.inter_users_dense_shape = np.array([self.num_items, self.num_users]).astype(np.int64)

        self.u_input = data_dict['u_input']
        self.f_input = data_dict['f_input']
        self.i_input = data_dict['i_input']
        self.seg_in_friends = data_dict['seg_in_friends']
        self.ufi_att = data_dict['ufi_att']
        self.int_att = data_dict['int_att']
        self.social_att_idx = data_dict['social_att_idx']
        self.social_att = data_dict['social_att']
        self.ufi_indices = np.concatenate([np.expand_dims(self.seg_in_friends, 1), np.expand_dims(self.i_input, 1)], 1)
        self.ufi_shape = np.array([np.max(self.seg_in_friends)+1, self.num_items]).astype(np.int64)

    def initializeNodes(self):
        super(disgcn, self).initializeNodes()
        self.cor_idx = tfv1.placeholder("int32", [None])
        self.ufi_u = tfv1.placeholder("int32", [None])
        self.ufi_f = tfv1.placeholder("int32", [None])
        self.ufi_g = tfv1.placeholder("int32", [None, self.num_negatives, 1])
        self.ufi_i = tfv1.placeholder("int32", [None])
        self.ufi_j = tfv1.placeholder("int32", [None, self.num_negatives, 1])

        self.user_social_embedding = tf.Variable(
                tf.random.normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_social_embedding')
        self.item_social_embedding = tf.Variable(
                tf.random.normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_social_embedding')

        # pre_W = np.load(os.path.join(os.getcwd(), 'embedding', self.conf.data_name, self.conf.pre_w), encoding='latin1')
        # self.W_ufi = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W0', use_bias=True, kernel_initializer=tf.constant_initializer(pre_W['W0_kernel']), bias_initializer=tf.constant_initializer(pre_W['W0_bias'])), \
        #                 tfv1.layers.Dense(1, name='W1', use_bias=True, kernel_initializer=tf.constant_initializer(pre_W['W1_kernel']), bias_initializer=tf.constant_initializer(pre_W['W1_bias']))]
        self.W = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name=f'W_{k}', use_bias=True) for k in range(self.conf.num_layers)]
        # self.bn_int = [tfv1.layers.BatchNormalization(name=f'bn_int_{k}') for k in range(self.conf.num_layers)]
        # self.bn_social = [tfv1.layers.BatchNormalization(name=f'bn_social_{k}') for k in range(self.conf.num_layers)]
        
    def construct_social_neighbors_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.social_neighbors_indices_input,
            values = tf.squeeze(sp_value),
            dense_shape=self.social_neighbors_dense_shape
        )

    def construct_consumed_items_sparse_matrix(self, sp_value):
        return tf.SparseTensor(
            indices = self.consumed_items_indices_input, 
            values = tf.squeeze(sp_value),
            dense_shape=self.consumed_items_dense_shape
        )

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y

    def Y(self, u, f, i, keepdims=True):
        # u, f, i = tf.math.l2_normalize(u, -1), tf.math.l2_normalize(f, -1), tf.math.l2_normalize(i, -1)
        return tf.reduce_sum(u*i, -1, keepdims=keepdims) + tf.reduce_sum(f*i, -1, keepdims=keepdims) + tf.reduce_sum(u*f, -1, keepdims=keepdims)
        # return tf.reduce_sum(tf.math.maximum(tf.math.maximum(u*i, f*i), u*f), -1, keepdims=keepdims)

        # result = self.W_ufi[1](self.W_ufi[0](tf.concat([u, f, i], -1)))
        # if not keepdims:
        #     result = tf.squeeze(result, -1)
        # return result

    def calc_ufi_att(self, user_social, item_social):
        f_emb, i_emb = tf.gather(user_social, self.f_input), tf.gather(item_social, self.i_input)
        ufi_att = self.Y(tf.gather(user_social, self.u_input), f_emb, i_emb, keepdims=False)
        ufi_matrix = tf.sparse.softmax(tf.SparseTensor(
            indices = self.ufi_indices,
            values = ufi_att,
            dense_shape = self.ufi_shape
        ))
        return tf.expand_dims(ufi_matrix.values, 1)

    def update_ufi_att(self, ufi_att_list):
        self.ufi_att_list_update = ufi_att_list

    def calc_att(self, user_int, user_social):
        u_int, f_int = tf.gather(user_int, np.squeeze(self.social_edges_user0)), tf.gather(user_int, np.squeeze(self.social_edges_user1))
        int_att = tf.reduce_sum(u_int*f_int, -1, keepdims=False)
        int_matrix = tf.sparse.softmax(self.construct_social_neighbors_sparse_matrix(int_att))

        u_social, f_social = tf.gather(user_social, self.social_att_idx[:, 0]), tf.gather(user_social, self.social_att_idx[:, 1])
        social_att = tf.reduce_sum(u_social*f_social, -1, keepdims=False)
        social_matrix = tf.sparse.softmax(tf.SparseTensor(
            indices = self.social_att_idx,
            values = social_att,
            dense_shape=self.social_neighbors_dense_shape
        )
        )
        return tf.expand_dims(int_matrix.values, 1), tf.expand_dims(social_matrix.values, 1)

    def update_att(self, int_att_list, social_att_list):
        self.int_att_list_update = int_att_list
        self.social_att_list_update = social_att_list

    def prop(self, user_int, user_social, item_social, layer):
        # user_int_from_friends = tf.sparse.sparse_dense_matmul(self.social_matrix, user_int)
        # next_user_int = user_int_from_friends
        next_user_int = tf.math.unsorted_segment_sum(tf.gather(user_int, np.squeeze(self.social_edges_user1))*self.int_att_list_update[layer], np.squeeze(self.social_edges_user0), self.num_users)
        
        # self.W[layer](f_emb*i_emb)+f_emb
        f_emb, i_emb = tf.gather(user_social, self.f_input), tf.gather(item_social, self.i_input)
        uf_prop = tf.math.segment_sum((self.W[layer](f_emb*i_emb)+f_emb)*self.ufi_att_list_update[layer], self.seg_in_friends)
        next_user_social = tf.math.unsorted_segment_sum(uf_prop*self.social_att_list_update[layer], self.social_att_idx[:, 0], self.num_users)
        #  + tf.sparse.sparse_dense_matmul(self.social_matrix, user_social)

        return next_user_int, next_user_social, uf_prop

    def calc_social_loss(self, user_social_list, item_social_list):
        # ufi_neg_i, ufi_neg_f, reg_loss = 0, 0, 0
        social_loss = 0
        for user_social, item_social in zip(user_social_list, item_social_list):
            emb_u, emb_f, emb_i = tf.gather(user_social, self.ufi_u), \
                                    tf.gather(user_social, self.ufi_f), \
                                    tf.gather(item_social, self.ufi_i)
            emb_j = tf.gather_nd(item_social, self.ufi_j)
            emb_g = tf.gather_nd(user_social, self.ufi_g)
            ufi_pos = self.Y(emb_u, emb_f, emb_i)
            ufi_neg_i = self.Y(tf.expand_dims(emb_u, 1), tf.expand_dims(emb_f, 1), emb_j, keepdims=False)
            # tile_num = tf.constant([1, self.num_negatives, 1], tf.int32)
            # ufi_neg_i = self.Y(tf.tile(tf.expand_dims(emb_u, 1), tile_num), tf.tile(tf.expand_dims(emb_f, 1), tile_num), emb_j, keepdims=False)
            ufi_neg_f = self.Y(tf.expand_dims(emb_u, 1), emb_g, tf.expand_dims(emb_i, 1), keepdims=False)
            # ufi_neg_f = self.Y(tf.tile(tf.expand_dims(emb_u, 1), tile_num), emb_g, tf.tile(tf.expand_dims(emb_i, 1), tile_num), keepdims=False)
            # reg_loss += self.regloss([emb_u, emb_f, emb_i])+self.regloss([emb_j, emb_g])/self.num_negatives
        # social_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(ufi_neg_i-ufi_pos), -1)) + tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(ufi_neg_f-ufi_pos), -1)) +\
            social_loss += tf.reduce_sum(tf.nn.softplus(tf.reduce_sum(ufi_neg_i-ufi_pos, -1))) + tf.reduce_sum(tf.nn.softplus(tf.reduce_sum(ufi_neg_f-ufi_pos, -1))) +\
                                self.conf.sreg*(self.regloss([emb_u, emb_f, emb_i])+self.regloss([emb_j, emb_g])/self.num_negatives)
                                # self.conf.sreg*reg_loss/len(user_social_list)
        return social_loss/len(user_social_list)

    def constructTrainGraph(self):
        # self.social_matrix = self.construct_social_neighbors_sparse_matrix(self.social_neighbors_values_input)
        self.consumed_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)
        user_int_from_items = tf.sparse.sparse_dense_matmul(self.consumed_matrix, self.item_embedding)
        user_social_from_items = tf.sparse.sparse_dense_matmul(self.consumed_matrix, self.item_social_embedding)

        self.ufi_att_list_update = [self.ufi_att for _ in range(self.conf.num_layers)]
        self.int_att_list_update = [self.int_att for _ in range(self.conf.num_layers)]
        self.social_att_list_update = [self.social_att for _ in range(self.conf.num_layers)]
        user_int, user_social = self.user_embedding, self.user_social_embedding
        user_int_list, user_social_list, uf_prop_list = [self.user_embedding], [self.user_social_embedding], []
        for k in range(self.conf.num_layers):
            user_int, user_social, uf_prop = self.prop(user_int, user_social, self.item_social_embedding, k)
            # user_int, user_social = tf.math.l2_normalize(user_int, 1), tf.math.l2_normalize(user_social, 1)
            user_int_list.append(user_int)
            user_social_list.append(user_social)
            # user_int, user_social = self.convertDistribution(user_int), self.convertDistribution(user_social)
            uf_prop_list.append(uf_prop)
            # user_int, user_social = self.bn_int[k](user_int), self.bn_social[k](user_social)
        user_int = tf.add_n(user_int_list)/(self.conf.num_layers+1) + user_int_from_items
        user_social = tf.add_n(user_social_list)/(self.conf.num_layers+1) + user_social_from_items
        self.ufi_att_list = [self.calc_ufi_att(us, self.item_social_embedding) for us in user_social_list[:self.conf.num_layers]]
        self.int_att_list, self.social_att_list = [], []
        for ui, us in zip(user_int_list[:self.conf.num_layers], uf_prop_list):
            int_att, social_att = self.calc_att(ui, us)
            self.int_att_list.append(int_att)
            self.social_att_list.append(social_att)

        user_embedding = tf.concat([user_int, user_social], 1)
        item_embedding = tf.concat([self.item_embedding, self.item_social_embedding], 1)
        # W_reg = 0
        # for W_k in self.W:
        #     W_reg += (tf.reduce_sum(tf.square(W_k.kernel)) + tf.reduce_sum(tf.square(W_k.bias)))
        self.loss = self.BPRloss(user_embedding, item_embedding)
        cor_social_embedding = tf.gather(tf.concat([self.user_social_embedding, self.item_social_embedding], 0), self.cor_idx)
        cor_interest_embedding = tf.gather(tf.concat([self.user_embedding, self.item_embedding], 0), self.cor_idx)
        self.cor_loss = self._create_distance_correlation(cor_social_embedding, cor_interest_embedding)
        self.loss += self.beta*self.cor_loss
        self.prediction = self.predict(user_embedding, item_embedding)
        # self.prediction = self.predict(user_int, self.item_embedding)

        self.social_loss = self.calc_social_loss(user_social_list, [self.item_social_embedding for _ in range(self.conf.num_layers+1)])
        # self.social_loss = self.calc_social_loss([user_social], [self.item_social_embedding])

        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.social_opt = tfv1.train.AdamOptimizer(self.conf.social_lr).minimize(self.social_loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['user_social_embedding'] = self.user_social_embedding
        variables_dict['item_social_embedding'] = self.item_social_embedding
        for k, W in enumerate(self.W):
            variables_dict[f'W_{k}/kernel'] = W.kernel
            variables_dict[f'W_{k}/bias'] = W.bias
        # for k, (bn_int, bn_social) in enumerate(zip(self.bn_int, self.bn_social)):
        #     variables_dict[f'bn_int_{k}/gamma'] = bn_int.gamma
        #     variables_dict[f'bn_int_{k}/beta'] = bn_int.beta
        #     variables_dict[f'bn_social_{k}/gamma'] = bn_social.gamma
        #     variables_dict[f'bn_social_{k}/beta'] = bn_social.beta
        # for k, W in enumerate(self.W_ufi):
        #     variables_dict[f'W_ufi_{k}/kernel'] = W.kernel
        #     variables_dict[f'W_ufi_{k}/bias'] = W.bias
        self.saver = tfv1.train.Saver(variables_dict)

    def defineMap(self):
        super(disgcn, self).defineMap()
        tmp_mask = {self.ufi_u:'ufi_u', self.ufi_f:'ufi_f', self.ufi_i:'ufi_i', self.ufi_j:'ufi_j', self.ufi_g:'ufi_g'}
        self.map_dict['train_social'] = tmp_mask
        self.map_dict['train'].update({self.cor_idx:'cor_idx'})