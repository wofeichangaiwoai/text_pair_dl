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
        self.input_y= tf.placeholder(tf.int32, [None], name='input_y')
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

            # for idx in range(num_layers):
            g = f(_linear(input_, size, scope=("highway_lin_{0}".format(num_layers))))
            t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(num_layers))) + bias)
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
        # self.pool_combine=tf.concat([self.pool_front_flat,self.pool_behind_flat],axis=1)
        # # self.cos=cos_sim(self.pool_front_flat,self.pool_behind_flat)
        # self.cos=tf.stack([cos_sim(self.pool_front_flat, self.pool_behind_flat)], axis=1)
        self.pool_combine = tf.concat([self.pool_front_flat, self.pool_behind_flat], axis=1)




        #Fully Connected layer
        with tf.name_scope('fc'):
            W1=tf.Variable(tf.truncated_normal(shape=[num_filters_total,fc_hidden_size],
                                              stddev=0.1,dtype=tf.float32),name='W1')
            b1=tf.Variable(tf.constant(value=0.1,shape=[fc_hidden_size],dtype=tf.float32),
                          name='b1')
            self.fc_front=tf.nn.xw_plus_b(self.pool_front_flat,W1,b1)

            self.fc_bn_front=tf.layers.batch_normalization(self.fc_front,training=self.is_training)

            self.fc_out_front=tf.nn.relu(self.fc_bn_front,name='relu1')

            W2 = tf.Variable(tf.truncated_normal(shape=[num_filters_total, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name='W2')
            b2 = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32),
                            name='b2')
            self.fc_behind = tf.nn.xw_plus_b(self.pool_behind_flat, W2, b2)

            self.fc_bn_behind = tf.layers.batch_normalization(self.fc_behind, training=self.is_training)

            self.fc_out_behind = tf.nn.relu(self.fc_bn_behind, name='relu2')

        with tf.name_scope('highway'):
            self.highway_front=_highway_layer(self.fc_out_front,self.fc_out_front.get_shape()[1],
                                        num_layers=1,bias=0)
            self.highway_behind = _highway_layer(self.fc_out_behind, self.fc_out_behind.get_shape()[1],
                                                num_layers=2, bias=0)
        with tf.name_scope('dropout'):
            self.h_drop_front=tf.nn.dropout(self.highway_front,self.dropout_keep_prob)
            self.h_drop_behind = tf.nn.dropout(self.highway_behind, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W1=tf.Variable(tf.truncated_normal(shape=[fc_hidden_size,num_classes],
                                              stddev=0.1,dtype=tf.float32),name='W1')
            b1=tf.Variable(tf.constant(value=0.1,shape=[num_classes],dtype=tf.float32),
                          name='b1')
            self.logits_front=tf.nn.xw_plus_b(self.h_drop_front,W1,b1,name='logits')

            W2 = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')
            b2 = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32),
                            name='b2')
            self.logits_behind = tf.nn.xw_plus_b(self.h_drop_behind, W2, b2, name='logits')

            f_x1x2 = tf.reduce_sum(tf.multiply(self.logits_front, self.logits_behind), 1)
            norm_fx1 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_front), 1))
            norm_fx2 = tf.sqrt(tf.reduce_sum(tf.square(self.logits_behind), 1))
            self.Ew = f_x1x2 / (norm_fx1 * norm_fx2)

            self.scores=tf.sigmoid(self.Ew,name='scores')



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

            # losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(self.input_y, tf.float32),
            #                                                  logits=self.Ew)
            #
            # losses = tf.reduce_mean(tf.reduce_sum(losses, axis=0), name='sigmoid_losses')
            # l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
            #                      name='l2_losses') * l2_reg_lambda
            # self.loss = tf.add(losses, l2_losses, name='loss')
            l_1 = 0.25 * tf.square(1 -self.Ew)
            l_0 = tf.square(tf.maximum(self.Ew, 0))
            self.input_y=tf.to_float(self.input_y)
            self.loss = tf.reduce_sum(self.input_y * l_1 + (1 - self.input_y) * l_0)
            # self.loss = self.contrastive_loss(self.Ew, self.input_y)
























































