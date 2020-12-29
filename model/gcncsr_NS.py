import tensorflow as tf
import numpy as np
import os
from base_model import base_model

class gcncsr(base_model):
    def inputSupply(self, data_dict):
        self.dp_ui = tf.placeholder("float32", [1])
        self.dp_ui_layer = tf.layers.Dropout(tf.gather(self.dp_ui, 0))
        self.social_edges_user0 = data_dict['social_edges_user0']
        self.social_edges_user1 = data_dict['social_edges_user1']
        self.interaction_user_indices = data_dict['interaction_user_indices']
        self.interaction_item_indices = data_dict['interaction_item_indices']
        self.consumed_items_indices_input = np.concatenate([self.interaction_user_indices, self.interaction_item_indices], 1)
        self.consumed_items_values_input = data_dict['consumed_items_values_input']
        self.social_neighbors_indices_input = np.concatenate([self.social_edges_user0, self.social_edges_user1], 1)
        if self.conf.test:
            conf = self.conf
            save_path = './pretrain/{}/{}_{}_reg{}_lr{}_epoch0+{}_dim{}_{}_att.npy'.format(conf.data_name, conf.data_name, conf.model_name, conf.reg[0], conf.learning_rate[0], conf.epochs, conf.dimension, conf.test_name)
            self.social_neighbors_values_input = np.load(save_path)
        else:
            self.social_neighbors_values_input = data_dict['social_neighbors_values_input']
        self.social_neighbors_values_input = self.dp_ui_layer(self.social_neighbors_values_input)
        self.consumed_items_values_input = self.dp_ui_layer(self.consumed_items_values_input)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        
        self.social_neighbors_sparse_matrix = self.construct_social_neighbors_sparse_matrix(self.social_neighbors_values_input)
        self.consumed_items_sparse_matrix = self.construct_consumed_items_sparse_matrix(self.consumed_items_values_input)

        self.seg_idx = tf.constant(np.squeeze(self.social_edges_user0), dtype=tf.int32)

    def initializeNodes(self):
        super(gcncsr, self).initializeNodes()
        self.num_layers = self.conf.num_layers
        self.relation_mean = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], mean=0.0, stddev=0.01), name='relation_mean')
        self.relation_var = tf.Variable(tf.random_normal([self.conf.num_mem, self.dim], stddev=0.01), name='relation_var')
        self.m = tf.distributions.Normal(0.0, 1.0)
        self.W_z = [tf.layers.Dense(int((self.dim*2+self.conf.num_mem)/2), activation=tf.nn.leaky_relu, name='W_z', use_bias=True), tf.layers.Dense(self.conf.num_mem, activation=tf.nn.leaky_relu, name='W_z', use_bias=True)]
        self.W1 = [tf.layers.Dense(self.dim, activation=tf.nn.leaky_relu, name='W1', use_bias=True) for _ in range(self.conf.num_layers)]
        self.W_fusion = [tf.layers.Dense(self.dim, name='W2', use_bias=True) for _ in range(self.conf.num_layers)]
    
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

    def edge_gen(self, u_emb, i_emb, norm=False):
        if norm:
            u_emb, i_emb = tf.math.l2_normalize(u_emb, axis=1), tf.math.l2_normalize(i_emb, axis=1)
        edges_preference_cat = tf.concat([u_emb, i_emb], 1)
        for k in range(len(self.W_z)):
            edges_preference_cat = self.W_z[k](edges_preference_cat)
        edges_preference_cat_softmax = tf.nn.softmax(edges_preference_cat)
        m_sample = self.m.sample([self.conf.num_mem, self.dim])
        M = self.relation_mean + self.relation_var*m_sample
        edges_preference = tf.matmul(edges_preference_cat_softmax, M)
        return edges_preference

    def constructTrainGraph(self):
        tmp_user_embedding = self.user_embedding
        user_embedding_from_item = tf.sparse_tensor_dense_matmul(self.consumed_items_sparse_matrix, self.item_embedding)
        user_embedding_last_list = [self.user_embedding]
        for k in range(self.num_layers):
            user_embedding_from_frined = tf.sparse_tensor_dense_matmul(self.social_neighbors_sparse_matrix, tmp_user_embedding)
            tmp_u0 = tf.gather_nd(tmp_user_embedding, self.social_edges_user0)
            tmp_u1 = tf.gather_nd(tmp_user_embedding, self.social_edges_user1)
            edge = self.edge_gen(tmp_u0, tmp_u1, k)
            sp_value = tf.expand_dims(self.social_neighbors_sparse_matrix._values, 1)
            user_embedding_from_edge = tf.unsorted_segment_sum(sp_value*self.W_fusion[k](tf.concat([tmp_u1, edge], 1)), self.seg_idx, num_segments=self.conf.num_users)
            tmp_user_embedding = self.W1[k](user_embedding_from_frined + user_embedding_from_edge)
            # tf.nn.leaky_relu(self.W1[k](user_embedding_from_frined) + self.W2[k](user_embedding_from_edge))
            user_embedding_last_list.append(tf.math.l2_normalize(tmp_user_embedding, axis=1))
            tmp_user_embedding = self.convertDistribution(tmp_user_embedding)
        user_embedding_last = tf.add_n(user_embedding_last_list)/(1+self.num_layers) + user_embedding_from_item
        item_embedding_last = self.item_embedding

        latest_user_embedding_latent = tf.gather_nd(user_embedding_last, self.user_input)
        latest_item_latent = tf.gather_nd(item_embedding_last, self.item_input)
        latest_item_neg_latent = tf.gather_nd(item_embedding_last, self.item_neg_input)

        W_loss = 0
        for k in range(self.num_layers):
            W_loss += (tf.reduce_sum(tf.square(self.W1[k].kernel)) + tf.reduce_sum(tf.square(self.W1[k].bias)))
            W_loss += (tf.reduce_sum(tf.square(self.W_fusion[k].kernel)) + tf.reduce_sum(tf.square(self.W_fusion[k].bias)))
        W_z_loss = 0
        for k in range(len(self.W_z)):
            W_z_loss += (tf.reduce_sum(tf.square(self.W_z[k].kernel)) + tf.reduce_sum(tf.square(self.W_z[k].bias)))
        m_loss = tf.reduce_sum(tf.square(self.relation_mean))
        v_loss = tf.reduce_sum(tf.square(self.relation_var))
        mem_loss = (W_z_loss + m_loss + v_loss)
        W_loss += mem_loss

        interaction_BPR_vector = tf.multiply(latest_user_embedding_latent, latest_item_latent - latest_item_neg_latent)
        self.prediction = tf.sigmoid(tf.matmul(latest_user_embedding_latent, tf.transpose(item_embedding_last)))
        self.reg_loss = tf.add_n([tf.reduce_sum(tf.square(latest_user_embedding_latent)), tf.reduce_sum(tf.square(latest_item_latent)), tf.reduce_sum(tf.square(latest_item_neg_latent)), W_loss])
        self.BPR_loss = -tf.reduce_sum(tf.log(tf.sigmoid(tf.reduce_sum(interaction_BPR_vector, 1, keepdims=True))))
        self.loss = self.reg*self.reg_loss + self.BPR_loss
        self.Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.opt = self.Adam.minimize(self.loss)
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
        for k in range(self.conf.num_layers):
            variables_dict['W1_{}/kernel'.format(k)] = self.W1[k].kernel
            variables_dict['W1_{}/bias'.format(k)] = self.W1[k].bias
            variables_dict['W_fusion_{}/kernel'.format(k)] = self.W_fusion[k].kernel
            variables_dict['W_fusion_{}/bias'.format(k)] = self.W_fusion[k].bias
        variables_dict['Adam'] = self.Adam
        self.saver = tf.train.Saver(variables_dict)