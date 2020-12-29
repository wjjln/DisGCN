import tensorflow as tf
import numpy as np
from base_model import base_model

class neumf(base_model):
    def inputSupply(self, data_dict):
        pass

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1])
        self.user_input = tf.placeholder("int32", [None, 1])
        self.item_neg_input = tf.placeholder("int32", [None, 1])
        self.num_layers = len(self.conf.fc)

        self.user_embedding_GMF = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding_GMF')
        self.item_embedding_GMF = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding_GMF')
        self.user_embedding_mlp = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding_mlp')
        self.item_embedding_mlp = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding_mlp')
        
        self.mlp_layer = [tf.layers.Dense(self.conf.fc[k], activation=tf.nn.leaky_relu, name='mlp_layer', use_bias=True) for k in range(self.num_layers)]
        self.h_GMF = tf.layers.Dense(\
                1, name='h_GMF', use_bias=False)
        self.h_mlp = tf.layers.Dense(\
                1, name='h_mlp', use_bias=False)

        self.emb_loss = 0

    def inference(self, item_input):
        i_emb_GMF = tf.gather_nd(self.item_embedding_GMF, item_input)
        i_emb_mlp = tf.gather_nd(self.item_embedding_mlp, item_input)
        self.emb_loss += tf.add_n([tf.reduce_sum(tf.square(i_emb_mlp)), tf.reduce_sum(tf.square(i_emb_GMF))])
        emb_GMF = self.u_emb_GMF*i_emb_GMF
        emb_mlp = tf.concat([self.u_emb_mlp, i_emb_mlp], 1)
        for k in range(self.num_layers):
            emb_mlp = self.mlp_layer[k](emb_mlp)
        return 0.5*(self.h_GMF(emb_GMF) + self.h_mlp(emb_mlp))

    def constructTrainGraph(self):
        self.u_emb_GMF = tf.gather_nd(self.user_embedding_GMF, self.user_input)
        self.u_emb_mlp = tf.gather_nd(self.user_embedding_mlp, self.user_input)
        self.emb_loss += tf.add_n([tf.reduce_sum(tf.square(self.u_emb_mlp)), tf.reduce_sum(tf.square(self.u_emb_GMF))])
        score_p = self.inference(self.item_input)
        score_n = self.inference(self.item_neg_input)
        self.prediction = score_p
        W_loss = 0
        W_loss += tf.reduce_sum(tf.square(self.h_GMF.kernel))
        for k in range(self.num_layers):
            W_loss += tf.add_n([tf.reduce_sum(tf.square(self.mlp_layer[k].kernel)), tf.reduce_sum(tf.square(self.mlp_layer[k].bias))])
        W_loss += tf.reduce_sum(tf.square(self.h_mlp.kernel))
        # self.loss = -(tf.reduce_sum(tf.log(score_p)) + tf.reduce_sum(tf.log(1-score_n))) + self.reg*self.reg_loss
        self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(score_p-score_n))) + self.reg*self.emb_loss + self.conf.w_reg*W_loss
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        variables_dict_GMF = {}
        variables_dict_GMF['user_embedding_GMF'] = self.user_embedding_GMF
        variables_dict_GMF['item_embedding_GMF'] = self.item_embedding_GMF
        variables_dict_GMF['h_GMF/kernel'] = self.h_GMF.kernel
        self.saver_GMF = tf.train.Saver(variables_dict_GMF)

        variables_dict_mlp = {}
        variables_dict_mlp['user_embedding_mlp'] = self.user_embedding_mlp
        variables_dict_mlp['item_embedding_mlp'] = self.item_embedding_mlp
        for k in range(self.num_layers):
            variables_dict_mlp['mlp_layer_{}/kernel'.format(k)] = self.mlp_layer[k].kernel
            variables_dict_mlp['mlp_layer_{}/bias'.format(k)] = self.mlp_layer[k].bias
        variables_dict_mlp['h_mlp/kernel'] = self.h_mlp.kernel
        self.saver_mlp = tf.train.Saver(variables_dict_mlp)

        variables_dict = variables_dict_GMF.update(variables_dict_mlp)
        self.saver = tf.train.Saver(variables_dict)
        