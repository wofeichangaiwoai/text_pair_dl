# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 15:43
# @Author  : bo
# @File    : text_cnn.py
# @Software: PyCharm


import tensorflow as tf
from tensorflow import tanh
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm

class BatchNormLSTM(rnn.RNNCell):
    """Batch normalized LSTM (cf. http://arxiv.org/abs/1603.09025)"""

    def __init__(self,num_units,is_training=False,forget_bias=1.0,actrivation=tanh,reuse=None):
        "Initialize the BNLSTM cell"
        self._num_units=num_units
        self._is_training=is_training
        self._forget_bias=forget_bias
        self._activation=actrivation
        self._reuse=reuse

class LSTM(object):
    def __init__(self,sequence_length,num_classes,vocab_size,fc_hidden_size,
                 embedding_size,embedding_type,num_units,l2_reg_lambda=0.0,
                 pretrained_embedding=None):
        #placeholders for input,output,dropout_prob and training_tag
        self.input_x_front=tf.placeholder(tf.int32,[None,sequence_length],name='input_x_front')
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name='input_x_behind')
        self.input_y= tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob=tf.placeholder(tf.float32,name='dropout_keep_prob')
        self.is_training=tf.placeholder(tf.bool,name='is_training')



        self.global_step=tf.Variable(0,trainable=False,name='Global_Step')

        def _linear(input_,output_size,scope='SimpleLinear'):
            shape=input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]
            with tf.variable_scope(scope):
                W=tf.get_variable('W',[input_size,output_size],dtype=input_.dtype)
                b=tf.get_variable('b',[output_size],dtype=input_.dtype)
            return tf.nn.xw_plus_b(input_,W,b)

        def _highway_layer(input_,size,num_layers=1,bias=-2.0,f=tf.nn.relu):
            """
                        Highway Network (cf. http://arxiv.org/abs/1505.00387).
                        t = sigmoid(Wy + b)
                        z = t * g(Wy + b) + (1 - t) * y
                        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
                        """

            for idx in range(num_layers):
                g = f(_linear(input_, size, scope=("highway_lin_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(idx))) + bias)
                output = t * g + (1. - t) * input_
                input_ = output

            return output

        def max_pooling(lstm_out):
            height,width=int(lstm_out.get_shape()[1]),int(lstm_out.get_shape()[2])

            lstm_out=tf.expand_dims(lstm_out,-1)
            output=tf.nn.max_pool(lstm_out,ksize=[1,height,1,1],strides=[1,1,1,1],padding='VALID')
            output=tf.reshape(output,[-1,width])
            return output

        def get_feature(input_q,input_a,att_W):
            h_q,w=int(input_q.get_shape()[1]),int(input_q.get_shape()[2])
            h_a=int(input_a.get_shape()[1])

            output_q=max_pooling(input_q)

            reshape_q=tf.expand_dims(output_q,1)
            reshape_q=tf.tile(reshape_q,[1,h_a,1])
            reshape_q=tf.reshape(reshape_q,[-1,w])
            reshape_a=tf.reshape(input_a,[-1,w])

            M=tf.tanh(tf.add(tf.matmul(reshape_q,att_W['Wqm']),tf.matmul(reshape_a,att_W['Wam'])))
            M=tf.matmul(M,att_W['Wms'])

            S=tf.reshape(M,[-1,h_a])
            S=tf.nn.softmax(S)

            S_diag=tf.matrix_diag(S)
            attention_a=tf.matmul(S_diag,input_a)
            attention_a=tf.reshape(attention_a,[-1,h_a,w])

            output_a=max_pooling(attention_a)

            return tf.tanh(output_q),tf.tanh(output_a)



        #Embedding Layer
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            if pretrained_embedding is None:
                self.embeding=tf.Variable(tf.random_uniform([vocab_size,embedding_size],
                                                            minval=-1.0,maxval=1.0,dtype=tf.float32),
                                          trainable=True,name='embedding')

            else:
                if embedding_type==0:
                    self.embeding=tf.constant(pretrained_embedding,dtype=tf.float32,
                                              name='embedding')
                if embedding_type==1:
                    self.embeding=tf.Variable(pretrained_embedding,trainable=True,
                                              dtype=tf.float32,name='embedding')
            self.embedded_sentence_front=tf.nn.embedding_lookup(self.embeding,self.input_x_front)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embeding, self.input_x_behind)


            # self.embedded_sentence_expanded_front=tf.expand_dims(self.embedded_sentence_front,axis=-1)
            # self.embedded_sentence_expanded_behind = tf.expand_dims(self.embedded_sentence_behind, axis=-1)

        with tf.name_scope("biLSTM1"):
            cell_fw=tf.nn.rnn_cell.LSTMCell(num_units)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            if self.dropout_keep_prob is not None:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,output_keep_prob=self.dropout_keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,output_keep_prob=self.dropout_keep_prob)
            outputs_front,outstates=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.
                                                              embedded_sentence_front,dtype=tf.float32,scope='biLSTM1')
            outputs_behind, outstates = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.
                                                                       embedded_sentence_behind, dtype=tf.float32,
                                                                       scope='biLSTM1')
        #     self.outputs_front=outputs1
        #
        # with tf.name_scope("biLSTM2"):
        #     cell_fw2=tf.nn.rnn_cell.LSTMCell(num_units)
        #     # cell_bw2 = tf.nn.rnn_cell.LSTMCell(num_units)
        #     outputs2,outstates=tf.nn.dynamic_rnn(cell=cell_fw2,inputs=self.
        #                                                       embedded_sentence_behind,dtype=tf.float32,scope='biLSTM2')
        #     self.outputs_behind=outputs2

        #

        # self.lstm_out_front = tf.expand_dims(tf.concat(outputs_front, axis=2),-1)
        # self.lstm_out_behind = tf.expand_dims(tf.concat(outputs_behind, axis=2),-1)
        self.lstm_out_front = tf.concat(outputs_front,axis=2)
        self.lstm_out_behind=tf.concat(outputs_behind,axis=2)


        # self.pool_combine=tf.concat([self.lstm_out_front,self.lstm_out_behind],axis=1)
        # pooled_front_outputs = []
        # pooled_behind_outputs = []
        # for filter_size in [3,4,5]:
        #     with tf.name_scope('conv-filter{0}'.format(filter_size)):
        #         # Convolution layers
        #         filter_shape = [filter_size, embedding_size, 1, 128]
        #         W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1,
        #                                             dtype=tf.float32), name='W')
        #         b = tf.Variable(tf.constant(value=0.1, shape=[128], dtype=tf.float32),
        #                         name='b')
        #         conv_front = tf.nn.conv2d(
        #             self.lstm_out_front,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name='conv'
        #         )
        #
        #         conv_behind = tf.nn.conv2d(
        #             self.lstm_out_behind,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name='conv'
        #         )
        #
        #         conv_front_bn = tf.layers.batch_normalization(tf.nn.bias_add(conv_front, b), training=self.is_training)
        #         conv_behind_bn = tf.layers.batch_normalization(tf.nn.bias_add(conv_behind, b),
        #                                                        training=self.is_training)
        #
        #         conv_front_out = tf.nn.relu(conv_front_bn, name="relu_front")
        #         conv_behind_out = tf.nn.relu(conv_behind_bn, name="relu_behind")
        #
        #     with tf.name_scope('pool-filter{0}'.format(filter_size)):
        #         pooled_front = tf.nn.max_pool(
        #             conv_front_out,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name='pool_front'
        #         )
        #         pooled_behind = tf.nn.max_pool(
        #             conv_behind_out,
        #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name='pool_behind'
        #         )
        #     pooled_front_outputs.append(pooled_front)
        #     pooled_behind_outputs.append(pooled_behind)
        #
        # num_filters_total = 128 * 3
        #
        # self.pool_front_flat = tf.reshape(tf.concat(pooled_front_outputs, axis=3), shape=[-1, num_filters_total])
        # self.pool_behind_flat = tf.reshape(tf.concat(pooled_behind_outputs, axis=3), shape=[-1, num_filters_total])

        #
        # # shape of `lstm_out_front`: [batch_size, lstm_hidden_size * 2]
        # self.lstm_out_front = tf.reduce_mean(self.lstm_concat_front, axis=1)
        # self.lstm_out_behind = tf.reduce_mean(self.lstm_concat_behind, axis=1)
        #
        # # shape of `lstm_out_concat`: [batch_size, lstm_hidden_size * 2 * 2]
        # self.lstm_out_concat = tf.concat([self.lstm_out_front, self.lstm_out_behind], axis=1)

        with tf.name_scope("att_weight"):
            # attention params
            att_W = {
                'Wam': tf.Variable(tf.truncated_normal([2 * num_units, embedding_size], stddev=0.1)),
                'Wqm': tf.Variable(tf.truncated_normal([2 * num_units, embedding_size], stddev=0.1)),
                'Wms': tf.Variable(tf.truncated_normal([embedding_size, 1], stddev=0.1))
            }

            self.atten_a,self.atten_q=get_feature(self.lstm_out_front,self.lstm_out_behind,att_W)
            self.atten_combine=tf.concat([self.atten_q,self.atten_a],axis=1)

            self.atten_combine=self.atten_combine+tf.concat([self.lstm_out_front[:,-1,:],self.lstm_out_behind[:,-1,:]],axis=1)
        # self.pool_combine=tf.concat([self.pool_front_flat,self.pool_behind_flat],axis=1)
        #Fully Connected layer
        with tf.name_scope('fc'):
            W=tf.Variable(tf.truncated_normal(shape=[4*num_units,fc_hidden_size],
                                              stddev=0.1,dtype=tf.float32),name='W')
            b=tf.Variable(tf.constant(value=0.1,shape=[fc_hidden_size],dtype=tf.float32),
                          name='b')
            self.fc=tf.nn.xw_plus_b(self.atten_combine ,W,b)


            self.fc_bn=batch_norm(self.fc, is_training=self.is_training, trainable=True, updates_collections=None)

            self.fc_out=tf.nn.relu(self.fc_bn,name='relu')

        with tf.name_scope('highway'):
            self.highway=_highway_layer(self.fc_out,self.fc_out.get_shape()[1],
                                        num_layers=1,bias=0)
        with tf.name_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.highway,self.dropout_keep_prob)

        with tf.name_scope('output'):
            W=tf.Variable(tf.truncated_normal(shape=[fc_hidden_size,num_classes],
                                              stddev=0.1,dtype=tf.float32),name='W')
            b=tf.Variable(tf.constant(value=0.1,shape=[num_classes],dtype=tf.float32),
                          name='b')
            self.logits=tf.nn.xw_plus_b(self.h_drop,W,b,name='logits')
            self.scores=tf.sigmoid(self.logits,name='scores')

        with tf.name_scope('loss'):


            #print(self.input_y,self.logits)

            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.input_y,tf.float32),
            #                                                  logits=self.logits)
            # # losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,
            # #                                                logits=self.logits)
            # losses=tf.reduce_mean(tf.reduce_sum(losses,axis=1),name='sigmoid_losses')
            # l2_losses=tf.add_n([tf.nn.l2_loss(tf.cast(v,tf.float32))for v in tf.trainable_variables()],
            #                    name='l2_losses')*l2_reg_lambda
            # self.loss=tf.add(losses,l2_losses,name='loss')

            losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32),
                                                             logits=self.logits)

            losses = tf.reduce_mean(tf.reduce_sum(losses, axis=0), name='sigmoid_losses')
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name='l2_losses') * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name='loss')


























































