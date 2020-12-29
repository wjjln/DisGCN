import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import os
import numpy as np
class base_model(object):
    def __init__(self, conf, reg, learning_rate):
        self.conf = conf
        self.reg = reg
        self.learning_rate = learning_rate
        self.dim = self.conf.dimension
        self.num_users = self.conf.num_users
        self.num_items = self.conf.num_items
        self.num_negatives = self.conf.num_negatives
        self.batch_size = self.conf.training_batch_size
        if conf.model_name in ['disengcn', 'disbpr', 'disgcn']:
            self.beta = conf.beta
    
    def inputSupply(self, data_dict):
        pass

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

    def initializeNodes(self):
        tfv1.disable_eager_execution()
        self.item_input = tfv1.placeholder("int32", [None])
        self.user_input = tfv1.placeholder("int32", [None])
        self.item_neg_input = tfv1.placeholder("int32", [None, self.num_negatives, 1])
        if self.conf.pretrain_flag:
            pre_emb = np.load(os.path.join(os.getcwd(), 'embedding', self.conf.data_name, self.conf.pre_train), encoding='latin1')
            user_embedding, item_embedding = pre_emb['user_embedding'], pre_emb['item_embedding']
            self.user_embedding = tf.Variable(
                user_embedding/np.linalg.norm(user_embedding, axis=1, keepdims=True), name='user_embedding')
            self.item_embedding = tf.Variable(
                item_embedding/np.linalg.norm(item_embedding, axis=1, keepdims=True), name='item_embedding')
            if 'dis' in self.conf.model_name:
                user_social_embedding, item_social_embedding = pre_emb['user_social_embedding'], pre_emb['item_social_embedding']
                self.user_social_embedding = tf.Variable(
                    user_social_embedding/np.linalg.norm(user_social_embedding, axis=1, keepdims=True), name='user_social_embedding')
                self.item_social_embedding = tf.Variable(
                    item_social_embedding/np.linalg.norm(item_social_embedding, axis=1, keepdims=True), name='item_social_embedding')
                # user_embedding_all, item_embedding_all = np.concatenate([user_embedding, user_social_embedding], 1), np.concatenate([item_embedding, item_social_embedding], 1)
                # user_embedding_all, item_embedding_all = user_embedding_all/np.linalg.norm(user_embedding_all, axis=1, keepdims=True), item_embedding_all/np.linalg.norm(item_embedding_all, axis=1, keepdims=True)
                # self.user_embedding = tf.Variable(
                #     user_embedding_all[:, :self.dim], name='user_embedding')
                # self.item_embedding = tf.Variable(
                #     item_embedding_all[:, :self.dim], name='item_embedding')
                # self.user_social_embedding = tf.Variable(
                #     user_embedding_all[:, self.dim:], name='user_social_embedding')
                # self.item_social_embedding = tf.Variable(
                #     item_embedding_all[:, self.dim:], name='item_social_embedding')
        else:
            self.user_embedding = tf.Variable(
                tf.random.normal([self.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
            self.item_embedding = tf.Variable(
                tf.random.normal([self.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')
    
    def constructTrainGraph(self):
        pass

    def saveVariables(self):
        pass

    def predict(self, emb_u=None, emb_i=None):
        if emb_u is None:
            emb_u = self.user_embedding
        if emb_i is None:
            emb_i = self.item_embedding
        emb_u_gather = tf.gather(emb_u, self.user_input)
        self.prediction = tf.matmul(emb_u_gather, tf.transpose(emb_i))

    def BPRloss(self, emb_u=None, emb_i=None, reg=True):
        if emb_u is None:
            emb_u = self.user_embedding
        if emb_i is None:
            emb_i = self.item_embedding
        emb_u_gather = tf.gather(emb_u, self.user_input)
        emb_i_gather = tf.gather(emb_i, self.item_input)
        emb_j_gather = tf.gather_nd(emb_i, self.item_neg_input)
        BPR_vector = tf.multiply(tf.expand_dims(emb_u_gather, 1), -(tf.expand_dims(emb_i_gather, 1)-emb_j_gather))
        # tmp_bpr = tf.math.log(tf.nn.sigmoid(tf.reduce_sum(BPR_vector, axis=-1)))
        loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softplus(tf.reduce_sum(BPR_vector, axis=-1)), -1))
        if reg:
            loss += self.reg*(self.regloss([emb_u_gather, emb_i_gather])+self.regloss([emb_j_gather])/self.num_negatives)
        return loss

    def regloss(self, tensors):
        loss = 0
        for t in tensors:
            loss += tf.nn.l2_loss(t)
        return loss
        # tf.losses.get_regularization_loss())

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)

            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        n_samples = tf.dtypes.cast(tf.shape(X1)[0], tf.float32)

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples**2), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10) * n_samples/2
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def defineMap(self):
        from copy import copy
        map_dict = {}
        tmp = {
            self.user_input: 'USER_LIST', 
            self.item_input: 'ITEM_LIST', 
            self.item_neg_input: 'ITEM_NEG_INPUT'
        }
        map_dict['train'] = tmp
        
        map_dict['val'] = copy(tmp)

        map_dict['test'] = copy(tmp)

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST',
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction#, self.prediction_link]
        }

        self.map_dict = map_dict


