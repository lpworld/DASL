import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class WeightsOrthogonalityConstraint(tf.keras.constraints.Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis

    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = tf.keras.backend.transpose(w)
        if(self.encoding_dim > 1):
            m = tf.keras.backend.dot(tf.keras.backend.transpose(w), tf.Variable(w)) - tf.keras.backend.eye(self.encoding_dim)
            return self.weightage * tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(m)))
        else:
            m = tf.keras.backend.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)

class Model(object):
    def __init__(self, user_count, item_count, batch_size):
        hidden_size = 128
        memory_window = 10
        self.batch_size = batch_size
        self.memory_window = memory_window
        
        self.u_1 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i_1 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y_1 = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist_1 = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.hist_cross_1 = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        
        self.u_2 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i_2 = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.y_2 = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist_2 = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.hist_cross_2 = tf.placeholder(tf.int32, [batch_size, memory_window]) # [B, T]
        self.lr = tf.placeholder(tf.float64, [])
        
        user_emb_w_1 = tf.get_variable("user_emb_w_1", [user_count, hidden_size // 2])
        item_emb_w_1 = tf.get_variable("item_emb_w_1", [item_count, hidden_size // 2])
        user_b_1 = tf.get_variable("user_b_1", [user_count], initializer=tf.constant_initializer(0.0))
        item_b_1 = tf.get_variable("item_b_1", [item_count], initializer=tf.constant_initializer(0.0))
        user_emb_w_2 = tf.get_variable("user_emb_w_2", [user_count, hidden_size // 2])
        item_emb_w_2 = tf.get_variable("item_emb_w_2", [item_count, hidden_size // 2])
        user_b_2 = tf.get_variable("user_b_2", [user_count], initializer=tf.constant_initializer(0.0))
        item_b_2 = tf.get_variable("item_b_2", [item_count], initializer=tf.constant_initializer(0.0))

        item_emb_1 = tf.concat([tf.nn.embedding_lookup(item_emb_w_1, self.i_1),tf.nn.embedding_lookup(user_emb_w_1, self.u_1)], axis=1)
        item_b_1 = tf.gather(item_b_1, self.i_1)
        user_b_1 = tf.gather(user_b_1, self.u_1)
        h_emb_1 = tf.concat([tf.nn.embedding_lookup(item_emb_w_1, tf.slice(self.hist_1, [0,0], [batch_size,memory_window])),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w_1, self.u_1), 1), [1, memory_window, 1])], axis=2)
        h_emb_cross_1 = tf.concat([tf.nn.embedding_lookup(item_emb_w_2, tf.slice(self.hist_cross_1, [0,0], [batch_size,memory_window])),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w_1, self.u_1), 1), [1, memory_window, 1])], axis=2)
        item_emb_2 = tf.concat([tf.nn.embedding_lookup(item_emb_w_2, self.i_2),tf.nn.embedding_lookup(user_emb_w_2, self.u_2)], axis=1)
        item_b_2 = tf.gather(item_b_2, self.i_2)
        user_b_2 = tf.gather(user_b_2, self.u_2)
        h_emb_2 = tf.concat([tf.nn.embedding_lookup(item_emb_w_2, tf.slice(self.hist_2, [0,0], [batch_size,memory_window])),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w_2, self.u_2), 1), [1, memory_window, 1])], axis=2)
        h_emb_cross_2 = tf.concat([tf.nn.embedding_lookup(item_emb_w_1, tf.slice(self.hist_cross_2, [0,0], [batch_size,memory_window])),tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w_2, self.u_2), 1), [1, memory_window, 1])], axis=2)
        
        user_emb_2 = tf.nn.embedding_lookup(user_emb_w_2, self.u_2)
        user_emb_21 = tf.nn.embedding_lookup(user_emb_w_1, self.u_2)
        user_emb_2_map = tf.layers.dense(user_emb_2, hidden_size//2, kernel_regularizer=WeightsOrthogonalityConstraint(hidden_size//2))
        self.orthogonal_loss = tf.losses.cosine_distance(labels=user_emb_21, predictions=user_emb_2_map, dim=1)

        with tf.variable_scope('domain_1',reuse=tf.AUTO_REUSE):
            output_1, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb_1, dtype=tf.float32)
            output_cross_2, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb_cross_2, dtype=tf.float32)
            
        with tf.variable_scope('domain_2',reuse=tf.AUTO_REUSE):
            output_2, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb_2, dtype=tf.float32)
            output_cross_1, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb_cross_1, dtype=tf.float32)
            
        with tf.variable_scope('score_1',reuse=tf.AUTO_REUSE): 
            preference_1, _ = self.seq_attention(output_1, output_cross_1, hidden_size, memory_window)
            preference_1 = tf.nn.dropout(preference_1, 0.1)

            #Combine Long-Short-Term-User-Preferences
            concat_1 = tf.concat([preference_1, item_emb_1], axis=1)
            concat_1 = tf.layers.batch_normalization(inputs=concat_1)
            concat_1 = tf.layers.dense(concat_1, 80, activation=tf.nn.sigmoid, name='f1')
            concat_1 = tf.layers.dense(concat_1, 40, activation=tf.nn.sigmoid, name='f2')
            concat_1 = tf.layers.dense(concat_1, 1, activation=None, name='f3')
            concat_1 = tf.reshape(concat_1, [-1])

            #Estmation of user preference by combing different components
            self.logits_1 = item_b_1 + concat_1 + user_b_1 # [B]exp
            self.score_1 = tf.sigmoid(self.logits_1)
            
        with tf.variable_scope('score_2',reuse=tf.AUTO_REUSE): 
            preference_2, _ = self.seq_attention(output_2, output_cross_2, hidden_size, memory_window)
            preference_2 = tf.nn.dropout(preference_2, 0.1)

            #Combine Long-Short-Term-User-Preferences
            concat_2 = tf.concat([preference_2, item_emb_2], axis=1)
            concat_2 = tf.layers.batch_normalization(inputs=concat_2)
            concat_2 = tf.layers.dense(concat_2, 80, activation=tf.nn.sigmoid, name='f1')
            concat_2 = tf.layers.dense(concat_2, 40, activation=tf.nn.sigmoid, name='f2')
            concat_2 = tf.layers.dense(concat_2, 1, activation=None, name='f3')
            concat_2 = tf.reshape(concat_2, [-1])

            #Estmation of user preference by combing different components
            self.logits_2 = item_b_2 + concat_2 + user_b_2 # [B]exp
            self.score_2 = tf.sigmoid(self.logits_2)
            
        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_1,labels=self.y_1))
        self.loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_2,labels=self.y_2))
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        
        gradients1 = tf.gradients(self.loss1, trainable_params)
        clip_gradients1, _ = tf.clip_by_global_norm(gradients1, 1)
        self.train_op1 = self.opt.apply_gradients(zip(clip_gradients1, trainable_params), global_step=self.global_step)
        
        gradients2 = tf.gradients(self.loss2, trainable_params)
        clip_gradients2, _ = tf.clip_by_global_norm(gradients2, 1)
        self.train_op2 = self.opt.apply_gradients(zip(clip_gradients2, trainable_params), global_step=self.global_step)
        
        gradients_orth = tf.gradients(self.orthogonal_loss, trainable_params)
        clip_gradients_orth, _ = tf.clip_by_global_norm(gradients_orth, 1)
        self.train_op_orth = self.opt.apply_gradients(zip(clip_gradients_orth, trainable_params), global_step=self.global_step)

    def train_1(self, sess, uij, lr):
        loss, _ = sess.run([self.loss1, self.train_op1], feed_dict={
                self.u_1: uij[0],
                self.hist_1: uij[1],
                self.hist_cross_1: uij[2],
                self.i_1: uij[3],
                self.y_1: uij[4],
                self.lr: lr
                })
        return loss

    def train_2(self, sess, uij, lr):
        loss, _ = sess.run([self.loss2, self.train_op2], feed_dict={
                self.u_2: uij[0],
                self.hist_2: uij[1],
                self.hist_cross_2: uij[2],
                self.i_2: uij[3],
                self.y_2: uij[4],
                self.lr: lr
                })
        return loss
    
    def train_orth(self, sess, uij, lr):
        loss, _ = sess.run([self.orthogonal_loss, self.train_op_orth], feed_dict={
                self.u_2: uij,
                self.lr: lr
                })
        return loss
    
    def test(self, sess, uij_1, uij_2):
        score_1, score_2 = sess.run([self.score_1, self.score_2], feed_dict={
                self.u_1: uij_1[0],
                self.hist_1: uij_1[1],
                self.hist_cross_1: uij_1[2],
                self.i_1: uij_1[3],
                self.y_1: uij_1[4],
                self.u_2: uij_2[0],
                self.hist_2: uij_2[1],
                self.hist_cross_2: uij_2[2],
                self.i_2: uij_2[3],
                self.y_2: uij_2[4]
                })
        return (score_1, uij_1[4], uij_1[0]), (score_2, uij_2[4], uij_2[0])

    def seq_attention(self, inputs, inputs_cross, hidden_size, attention_size):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
        for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
        """
        # Trainable parameters
        attention_size = attention_size * 2
        input_concat = tf.concat([inputs, inputs_cross],axis=1)
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(input_concat, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        alphas = tf.slice(alphas, [0,0], [self.batch_size, self.memory_window])
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas
