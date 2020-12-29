import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
from model.base_model import base_model

class disbpr(base_model):

    def initializeNodes(self):
        super(disbpr, self).initializeNodes()
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
        # self.W = [tfv1.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W0', use_bias=True), tfv1.layers.Dense(1, name='W1', use_bias=True)]
        # self.W = tf.Variable(
        #         tf.random.normal([self.dim, self.dim], stddev=0.01), name='W')

    def Y(self, u, f, i, keepdims=True):

        # def cosine(e1, e2, keepdims=True):
        #     e1 = tf.math.l2_normalize(e1, -1)
        #     e2 = tf.math.l2_normalize(e2, -1)
        #     return tf.reduce_sum(e1*e2, -1, keepdims=keepdims)
        
        # return tf.reduce_sum(u*i, -1, keepdims=keepdims)*cosine(f, i, keepdims=keepdims)

        return tf.reduce_sum(u*i, -1, keepdims=keepdims) + tf.reduce_sum(f*i, -1, keepdims=keepdims)# + tf.reduce_sum(u*f, -1, keepdims=keepdims)

        # return tf.reduce_sum(tf.math.maximum(tf.math.maximum(u*i, f*i), u*f), -1, keepdims=keepdims)

        # return -tf.reduce_sum(tf.square((u-f)*i), -1, keepdims=keepdims)

        # result = self.W[1](self.W[0](tf.concat([u, f, i], -1)))
        # if not keepdims:
        #     result = tf.squeeze(result, -1)
        # return result

        # if keepdims:
        #     u = tf.einsum('ik, kl -> il', u, self.W)
        #     f = tf.einsum('ik, kl -> il', f, self.W)
        #     return tf.reduce_sum(tf.square(u+i-f), -1, keepdims=True)
        # else:
        #     u = tf.einsum('ijk, kl -> ijl', u, self.W)
        #     f = tf.einsum('ijk, kl -> ijl', f, self.W)
        #     return tf.reduce_sum(tf.square(u+i-f), -1, keepdims=False)
    
    def constructTrainGraph(self):
        user_embedding = tf.concat([self.user_embedding, self.user_social_embedding], 1)
        item_embedding = tf.concat([self.item_embedding, self.item_social_embedding], 1)
        self.loss = self.BPRloss(user_embedding, item_embedding)
        self.predict(user_embedding, item_embedding)
        cor_social_embedding = tf.gather(tf.concat([self.user_social_embedding, self.item_social_embedding], 0), self.cor_idx)
        cor_interest_embedding = tf.gather(tf.concat([self.user_embedding, self.item_embedding], 0), self.cor_idx)
        self.cor_loss = self._create_distance_correlation(cor_social_embedding, cor_interest_embedding)
        self.loss += self.beta*self.cor_loss
        emb_u, emb_f, emb_i = tf.gather(self.user_social_embedding, self.ufi_u), \
                                tf.gather(self.user_social_embedding, self.ufi_f), \
                                tf.gather(self.item_social_embedding, self.ufi_i)
        emb_j = tf.gather_nd(self.item_social_embedding, self.ufi_j)
        emb_g = tf.gather_nd(self.user_social_embedding, self.ufi_g)
        ufi_pos_i = self.Y(emb_u, emb_f, emb_i)
        ufi_neg_i = self.Y(tf.expand_dims(emb_u, 1), tf.expand_dims(emb_f, 1), emb_j, keepdims=False)
        # tile_num = tf.constant([1, self.num_negatives, 1], tf.int32)
        # ufi_neg_i = self.Y(tf.tile(tf.expand_dims(emb_u, 1), tile_num), tf.tile(tf.expand_dims(emb_f, 1), tile_num), emb_j, keepdims=False)
        ufi_pos_f = self.Y(emb_u, emb_i, emb_f)
        ufi_neg_f = self.Y(tf.expand_dims(emb_u, 1), tf.expand_dims(emb_i, 1), emb_g, keepdims=False)
        # ufi_neg_f = self.Y(tf.tile(tf.expand_dims(emb_u, 1), tile_num), tf.tile(tf.expand_dims(emb_i, 1), tile_num), emb_g, keepdims=False)
        self.social_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(ufi_neg_i-ufi_pos_i), -1)) + tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(ufi_neg_f-ufi_pos_f), -1))
                            # self.reg*(self.regloss([emb_u, emb_f, emb_i])+self.regloss([emb_j, emb_g])/self.num_negatives)

        self.opt = tfv1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.social_opt = tfv1.train.AdamOptimizer(self.conf.social_lr).minimize(self.social_loss)
        self.init = tfv1.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['user_social_embedding'] = self.user_social_embedding
        variables_dict['item_social_embedding'] = self.item_social_embedding
        # for idx, W in enumerate(self.W):
        #     variables_dict[f'W{idx}/kernel'] = W.kernel
        #     variables_dict[f'W{idx}/bias'] = W.bias
        self.saver = tfv1.train.Saver(variables_dict)

    def defineMap(self):
        super(disbpr, self).defineMap()
        tmp_mask = {self.ufi_u:'ufi_u', self.ufi_f:'ufi_f', self.ufi_i:'ufi_i', self.ufi_j:'ufi_j', self.ufi_g:'ufi_g'}
        # tmp_eva = {self.ufi_u:'eva_ufi_u', self.ufi_f:'eva_ufi_f', self.ufi_i:'eva_ufi_i', self.ufi_idx:'eva_ufi_idx'}
        self.map_dict['train_social'] = tmp_mask
        self.map_dict['train'].update({self.cor_idx:'cor_idx'})
        # self.map_dict['eva'].update(tmp_eva)