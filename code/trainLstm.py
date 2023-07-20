# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 17:56
# @Author  : bo
# @File    : train_cnn.py
# @Software: PyCharm

#20191109修改

import os
import time
import logging
import numpy as np
import tensorflow as tf
from lstm import LSTM
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import xconfig


from gensim.models import word2vec
logging.info("✔︎The format of your input is legal, now loading to next step...")
logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))

import fasttext
word2vec_file =xconfig.word2vec_file

model=fasttext.load_model(word2vec_file)
w2v_model=model

vocab=dict([(k,v) for (v,k) in enumerate(model.get_words())])

def train_cnn():
    logger.info("loading data ...")
    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data_and_labels(xconfig.training_data_file, xconfig.num_classes,
                                         vocab, data_aug_flag=False)

    logger.info("Validation data processing...")
    val_data=dh.load_data_and_labels(xconfig.validation_data_file,
                                     xconfig.num_classes,vocab,data_aug_flag=False)

    logger.info("Recommended padding Sequence length is: {0}".format(xconfig.pad_seq_len))

    logger.info("Training data padding...")
    x_train_front,x_train_behind,y_train=dh.pad_data(train_data,xconfig.pad_seq_len)

    logger.info("Validation data padding...")
    x_val_front, x_val_behind,y_val = dh.pad_data(val_data, xconfig.pad_seq_len)

    # VOCAB_SIZE,pretrained_word2vec_matrix=dh.load_word2vec_matrix(xconfig.embedding_dim)
    VOCAB_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(xconfig.embedding_dim, word2vec_file, w2v_model,
                                                                     vocab)

    #Build a graph and cnn object
    with tf.Graph().as_default():
        # session_conf=tf.compat.v1.ConfigProto(
        #     # allow_soft_placemnt=xconfig.allow_soft_placement,
        #     # log_device_placement=xconfig.log_device_placement,
        #
        #     # log_device_placement=False
        # )
        session_conf = tf.ConfigProto(
            allow_soft_placement=xconfig.allow_soft_placement,
            log_device_placement=xconfig.log_device_placement)
        session_conf.gpu_options.allow_growth=xconfig.gpu_options_allow_growth
        sess=tf.Session(config=session_conf)

        # print('pretrained_word2vec_matrix:',pretrained_word2vec_matrix)
        with sess.as_default():
            cnn=LSTM(
                sequence_length=xconfig.pad_seq_len,
                num_classes=xconfig.num_classes,
                vocab_size=VOCAB_SIZE,
                fc_hidden_size=xconfig.fc_hidden_size,
                embedding_size=xconfig.embedding_dim,
                embedding_type=xconfig.embedding_type,
                # filter_sizes=list(map(int, xconfig.filter_sizes.split(','))),
                num_units=64,
                l2_reg_lambda=xconfig.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)




            #Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate=tf.train.exponential_decay(
                    learning_rate=xconfig.learning_rate,
                    global_step=cnn.global_step,
                    decay_steps=xconfig.decay_steps,
                    decay_rate=xconfig.decay_rate,
                    staircase=True
                )
                optimizer=tf.train.AdamOptimizer(learning_rate)
                grads,vars=zip(*optimizer.compute_gradients(cnn.loss))
                grads,_=tf.clip_by_global_norm(grads,clip_norm=xconfig.norm_ratio)
                train_op=optimizer.apply_gradients(zip(grads,vars),
                                                   global_step=cnn.global_step,name="train_op")

            #Keep track of gradient values and sparsity(optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

            #Output directory for models and summaries
            timestamp=str(int(time.time()))
            out_dir=os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
            logger.info("Writing to {0}\n".format(out_dir))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            #Train summaries
            train_summary_op=tf.compat.v1.summary.merge([loss_summary,grad_summaries_merged])
            train_summary_dir=os.path.join(out_dir,"summaries","train")
            train_summary_writer=tf.compat.v1.summary.FileWriter(train_summary_dir,sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)


            best_saver=cm.BestCheckpointSaver(save_dir=best_checkpoint_dir,num_to_keep=3,
                                               maximize=True)


            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            current_step=sess.run(cnn.global_step)

            def train_step(x_batch_front,x_batch_behind,y_batch):
                "A single training step"
                feed_dict={
                    cnn.input_x_front:x_batch_front,
                    cnn.input_x_behind:x_batch_behind,
                    cnn.input_y:y_batch,
                    cnn.dropout_keep_prob:xconfig.dropout_keep_prob,
                    cnn.is_training:1
                }
                _,step,summaries,loss=sess.run(
                    [train_op,cnn.global_step,train_summary_op,cnn.loss],
                    feed_dict
                )
                logger.info("step {0}:loss {1:g}".format(step,loss))


            def validation_step(x_val_front,x_val_behond,y_val,writer=None):
                "Evaluates model on a validation set"
                batches_validation=dh.batch_iter(list(zip(x_val_front,x_val_behond,y_val)),
                                                 xconfig.batch_size,1)

                #Predict classes by threshold or topk('ts':threshold;'tk':topk)
                eval_counter,eval_loss=0,0.0
                true_onehot_labels=[]
                predicted_onehot_scores=[]
                predicted_onehot_labels=[]
                # predicted_onehot_labels_tk=[[] for _ in range(xconfig.top_num)]

                for batch_validation in batches_validation:
                    x_batch_val_front,x_batch_val_behind,y_batch_val=zip(*batch_validation)
                    feed_dict={
                        cnn.input_x_front:x_batch_val_front,
                        cnn.input_x_behind: x_batch_val_behind,
                        cnn.input_y:y_batch_val,
                        cnn.dropout_keep_prob:1.0,
                        cnn.is_training:False
                    }
                    step,summaries,scores,cur_loss=sess.run(
                        [cnn.global_step,validation_summary_op,
                         cnn.scores,cnn.loss],feed_dict
                    )

                    #Prepare  for calculating metrics
                    for i in y_batch_val:
                        true_onehot_labels.append(i)
                    for j in scores:
                        predicted_onehot_scores.append(j)

                    #Predict by threshold
                    batch_predicted_onehot_labels=dh.get_onehot_label(
                        scores=scores)

                    for k in batch_predicted_onehot_labels:
                        predicted_onehot_labels.append(k)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)



                eval_pre, eval_rec, eval_F, eval_auc, eval_prc = dh.evaluation_calculation(true_onehot_labels,
                                                                                predicted_onehot_labels,
                                                                                predicted_onehot_scores)

                every_label_pre, every_label_rec, every_label_F = dh.evaluation_every_label(true_onehot_labels,
                                                                                            predicted_onehot_labels,
                                                                                            predicted_onehot_scores)


                return eval_loss, eval_auc, eval_prc, eval_rec, eval_pre, eval_F,every_label_pre, every_label_rec, every_label_F

            #Generate batches
            batches_train=dh.batch_iter(
                list(zip(x_train_front,x_train_behind,y_train)),xconfig.batch_size,xconfig.num_epochs
            )
            num_batches_per_epoch=int(len(x_train_front-1)/xconfig.batch_size)+1

            #Training loop,for each batch...
            for batch_train in batches_train:
                x_batch_train_front, x_batch_train_behind,y_batch_train = zip(*batch_train)
                train_step(x_batch_train_front,x_batch_train_behind, y_batch_train)
                current_step = tf.train.global_step(sess, cnn.global_step)

                if current_step % xconfig.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, eval_rec, eval_pre, eval_F,every_label_pre, every_label_rec, every_label_F = \
                        validation_step(x_val_front,x_val_behind, y_val, writer=validation_summary_writer)

                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))

                    # Predict by threshold
                    logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                                .format(eval_pre, eval_rec, eval_F))
                    label_id = 0
                    for pre, rec, f1 in zip(every_label_pre, every_label_rec, every_label_F):
                        logger.info("Predict : label {0},Precision {1:g}, Recall {2:g}, F1 {3:g}"
                                    .format(label_id, round(pre, 4), round(rec, 4), round(f1, 4)))
                        label_id += 1

                    best_saver.handle(eval_prc, sess, current_step)

                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("✔︎ Epoch {0} has finished!".format(current_epoch))

        logger.info("✔︎ Done.")

if __name__ == '__main__':
    train_cnn()

















































































