# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 15:43
# @Author  : bo
# @File    : text_cnn.py
# @Software: PyCharm


import tensorflow as tf

class TextCNN(object):
    def __init__(self,sequence_length,num_classes,vocab_size,fc_hidden_size,
                 embedding_size,embedding_type,filter_sizes,num_filters,l2_reg_lambda=0.0,
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


            self.embedded_sentence_expanded_front=tf.expand_dims(self.embedded_sentence_front,axis=-1)
            self.embedded_sentence_expanded_behind = tf.expand_dims(self.embedded_sentence_behind, axis=-1)


        #Create a convolution and maxpool layer for each filter size

        def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
            # x2 => [batch, height, 1, width]
            # [batch, width, wdith] = [batch, s, s]

            # 作者论文中提出计算attention的方法 在实际过程中反向传播计算梯度时 容易出现NaN的情况 这里面加以修改
            # euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            # return 1 / (1 + euclidean)

            x1 = tf.transpose(tf.squeeze(x1, [-1]), [0, 2, 1])
            attention = tf.einsum("ijk,ikl->ijl", x1, tf.squeeze(x2, [-1]))
            return attention


        pooled_front_outputs=[]
        pooled_behind_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope('conv-filter{0}'.format(filter_size)):
                #Convolution layers
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W=tf.Variable(tf.truncated_normal(shape=filter_shape,stddev=0.1,
                                                  dtype=tf.float32),name='W')
                b=tf.Variable(tf.constant(value=0.1,shape=[num_filters],dtype=tf.float32),
                              name='b')
                conv_front=tf.nn.conv2d(
                    self.embedded_sentence_expanded_front,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv'
                )

                conv_behind = tf.nn.conv2d(
                    self.embedded_sentence_expanded_behind,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv'
                )



                conv_front_bn=tf.layers.batch_normalization(tf.nn.bias_add(conv_front,b),training=self.is_training)
                conv_behind_bn = tf.layers.batch_normalization(tf.nn.bias_add(conv_behind, b), training=self.is_training)

                conv_front_out=tf.nn.relu(conv_front_bn,name="relu_front")
                conv_behind_out = tf.nn.relu(conv_behind_bn, name="relu_behind")

                conv_front_out = tf.layers.conv2d(conv_front_out, num_filters, filter_size, padding='same', activation=tf.nn.relu)
                conv_front_out = tf.layers.batch_normalization(conv_front_out)
                conv_front_out = conv_front_out+conv_front

                conv_behind_out = tf.layers.conv2d(conv_behind_out, num_filters, filter_size, padding='same',
                                                  activation=tf.nn.relu)
                conv_behind_out = tf.layers.batch_normalization(conv_behind_out)
                conv_behind_out = conv_behind_out + conv_behind


            with tf.name_scope('pool-filter{0}'.format(filter_size)):
                pooled_front=tf.nn.max_pool(
                    conv_front_out,
                    ksize=[1,sequence_length-filter_size+1,1,1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name='pool_front'
                )
                pooled_behind = tf.nn.max_pool(
                    conv_behind_out,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name='pool_behind'
                )
            pooled_front_outputs.append(pooled_front)
            pooled_behind_outputs.append(pooled_behind)

        num_filters_total=num_filters*len(filter_sizes)


        self.pool_front_flat=tf.reshape(tf.concat(pooled_front_outputs,axis=3),shape=[-1,num_filters_total])
        self.pool_behind_flat = tf.reshape(tf.concat(pooled_behind_outputs, axis=3), shape=[-1, num_filters_total])
        self.pool_combine=tf.concat([self.pool_front_flat,self.pool_behind_flat],axis=1)

        #Fully Connected layer
        with tf.name_scope('fc'):
            W=tf.Variable(tf.truncated_normal(shape=[2*num_filters_total,fc_hidden_size],
                                              stddev=0.1,dtype=tf.float32),name='W')
            b=tf.Variable(tf.constant(value=0.1,shape=[fc_hidden_size],dtype=tf.float32),
                          name='b')
            self.fc=tf.nn.xw_plus_b(self.pool_combine,W,b)

            self.fc_bn=tf.layers.batch_normalization(self.fc,training=self.is_training)

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
            self.scores=tf.nn.softmax(self.logits,name='scores')

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

            # losses = tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.input_y, tf.float32),
            #                                                  self.logits,2.0)
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
            # losses = tf.reduce_mean(tf.reduce_sum(losses, axis=0), name='sigmoid_losses')
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name='l2_losses') * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name='loss')


























































