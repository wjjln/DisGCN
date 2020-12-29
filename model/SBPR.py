import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class sbpr(base_model):
    def inputSupply(self, data_dict):
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.social_edges_user_neg = data_dict['social_edges_user_neg']
        
    def initializeNodes(self):
        super(sbpr, self).initializeNodes()
        self.user_social_loss_idx = tf.placeholder("int32", [None, 1])
        self.relation_mean = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension], mean=0.0, stddev=0.01), name='relation_mean')
        self.relation_var = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension], stddev=0.01), name='relation_var')
        self.m = tf.distributions.Normal(0.0, 1.0)
        self.W_z = [tf.layers.Dense(64, activation=tf.nn.leaky_relu, name='W_z', use_bias=True), tf.layers.Dense(self.conf.num_mem, activation=tf.nn.leaky_relu, name='W_z', use_bias=True)]
        self.Wr = tf.Variable(tf.random_normal([self.conf.num_mem, self.conf.dimension, self.conf.dimension], stddev=0.01), name='Wr')

    def edge_gen(self, u_emb, i_emb, batch=True, norm=True):
        if norm:
            u_emb, i_emb = tf.math.l2_normalize(u_emb, axis=1), tf.math.l2_normalize(i_emb, axis=1)
        edges_preference_cat = tf.concat([u_emb, i_emb], 1)
        for k in range(len(self.W_z)):
            edges_preference_cat = self.W_z[k](edges_preference_cat)
        edges_preference_cat_softmax = tf.nn.softmax(edges_preference_cat)
        m_sample = self.m.sample([self.conf.num_mem, self.conf.dimension])
        M = self.relation_mean + self.relation_var*m_sample
        edges_preference = tf.matmul(edges_preference_cat_softmax, M)
        # max_edge_idx = tf.expand_dims(tf.math.argmax(edges_preference_cat, 1), 1)
        # edges_preference = tf.gather_nd(M, max_edge_idx)
        if batch:
            Wr = tf.tensordot(edges_preference_cat_softmax, self.Wr, axes=1)
            # Wr = tf.squeeze(tf.gather_nd(self.Wr, max_edge_idx))
            return edges_preference, Wr
        else:
            return edges_preference

    def trans_infer(self, user_embedding, u0, u1):
        u0_emb, u1_emb = tf.gather_nd(user_embedding, u0), tf.gather_nd(user_embedding, u1)
        edge, Wr = self.edge_gen(u0_emb, u1_emb)
        u0_emb = tf.squeeze(tf.matmul(tf.expand_dims(u0_emb, 1), Wr))
        u1_emb = tf.squeeze(tf.matmul(tf.expand_dims(u1_emb, 1), Wr))
        return u0_emb, u1_emb, edge

    def constructTrainGraph(self):
        u_emb = tf.gather_nd(self.user_embedding, self.user_input)
        i_emb = tf.gather_nd(self.item_embedding, self.item_input)
        i_emb_n = tf.gather_nd(self.item_embedding, self.item_neg_input)

        u0 = tf.gather_nd(self.social_edges_user0, self.user_social_loss_idx)
        u1 = tf.gather_nd(self.social_edges_user1, self.user_social_loss_idx)
        # un = tf.gather_nd(self.social_edges_user_neg, self.user_social_loss_idx)
        u0_emb, u1_emb, edge_p = self.trans_infer(self.user_embedding, u0, u1)
        # u0_emb_n, un_emb, edge_n = self.trans_infer(self.user_embedding, u0, un)
        social_score_p = -tf.reduce_sum(tf.abs((u0_emb - u1_emb)*edge_p), 1)
        social_score_n = -tf.reduce_sum(tf.abs(u0_emb - u1_emb), 1)
        self.social_reg_loss = tf.add_n([tf.reduce_sum(tf.square(u0_emb)), tf.reduce_sum(tf.square(u1_emb))])
                                    # tf.reduce_sum(tf.square(u0_emb_n)), tf.reduce_sum(tf.square(un_emb))])
        W_z_loss = 0
        for k in range(len(self.W_z)):
            W_z_loss += (tf.reduce_sum(tf.square(self.W_z[k].kernel)) + tf.reduce_sum(tf.square(self.W_z[k].bias)))
        m_loss = tf.reduce_sum(tf.square(self.relation_mean))
        v_loss = tf.reduce_sum(tf.square(self.relation_var))
        Wr_loss = tf.reduce_sum(tf.square(self.Wr))
        mem_loss = (W_z_loss + m_loss + v_loss)
        self.social_reg_loss += mem_loss

        BPR_vector = tf.multiply(u_emb, i_emb - i_emb_n)
        self.prediction = tf.sigmoid(tf.matmul(u_emb, tf.transpose(self.item_embedding)))
        self.loss = self.reg*tf.add_n([tf.reduce_sum(tf.square(u_emb)), tf.reduce_sum(tf.square(i_emb)), tf.reduce_sum(tf.square(i_emb_n))])\
                    - tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(BPR_vector, axis=1))))
        self.social_loss = self.conf.social_reg*self.social_reg_loss - tf.reduce_sum(tf.log(tf.nn.sigmoid(social_score_p - social_score_n)))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.social_opt = tf.train.AdamOptimizer(self.conf.social_lr).minimize(self.social_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict = {}
        variables_dict['user_embedding'] = self.user_embedding
        variables_dict['item_embedding'] = self.item_embedding
        variables_dict['relation_mean'] = self.relation_mean
        variables_dict['relation_var'] = self.relation_var
        for k in range(len(self.W_z)):
            variables_dict['W_z_{}/kernel'.format(k)] = self.W_z[k].kernel
            variables_dict['W_z_{}/bias'.format(k)] = self.W_z[k].bias
        variables_dict['Wr'] = self.Wr
        self.saver = tf.train.Saver(variables_dict)

    # def defineMap(self):
    #     super(sbpr, self).defineMap()
    #     self.map_dict['train'][self.user_social_loss_idx] = 'USER_SOCIAL_LOSS_IDX'
    #     self.map_dict['test'][self.user_social_loss_idx] = 'USER_SOCIAL_LOSS_IDX'
    #     self.map_dict['val'][self.user_social_loss_idx] = 'USER_SOCIAL_LOSS_IDX'