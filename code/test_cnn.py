# -*- coding: utf-8 -*-
# @Time    : 2019-07-18 10:34
# @Author  : bo
# @File    : test.py
# @Software: PyCharm
#20191109修改

import os
import sys
import time
import numpy as np
import tensorflow as tf
import logging
from utils import checkmate as cm
from utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from utils import xconfig
from gensim.models import word2vec
# Parameters
# ==================================================
logging.info("✔︎The format of your input is legal, now loading to next step...")
logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()))


MODEL = xconfig.checkpoint_file


checkpoint_dir = 'runs/' + str(MODEL) + '/checkpoints/'
best_checkpoint_dir = 'runs/' + str(MODEL) + '/bestcheckpoints/'
SAVE_DIR = 'results/' + str(MODEL)
import fasttext
word2vec_file =xconfig.word2vec_file

model=fasttext.load_model(word2vec_file)
w2v_model=model

vocab=dict([(k,v) for (v,k) in enumerate(model.get_words())])

def test_cnn():
    """Test CNN model."""

    # Load data
    logger.info("✔︎ Loading data...")
    logger.info("Recommended padding Sequence length is: {0}".format(xconfig.pad_seq_len))

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data_and_labels(xconfig.test_data_file, xconfig.num_classes,
                                        vocab, data_aug_flag=False)

    # test_data_dic=dh.get_test_data(xconfig.data_file)

    logger.info("✔︎ Test data padding...")
    x_test_front, x_test_behind,y_test = dh.pad_data(test_data, xconfig.pad_seq_len)
    y_test_labels = test_data.labels

    # Load cnn model

    logger.info("✔︎ Loading best model...")
    checkpoint_file = cm.get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True)

    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=xconfig.allow_soft_placement,
            log_device_placement=xconfig.log_device_placement)
        session_conf.gpu_options.allow_growth = xconfig.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_front = graph.get_operation_by_name("input_x_front").outputs[0]
            input_x_behind=graph.get_operation_by_name("input_x_behind").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-cnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test_front,x_test_behind, y_test, y_test_labels)), xconfig.batch_size, 1, shuffle=False)

            test_counter, test_loss = 0, 0.0

            # Collect the predictions here
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            # Collect for calculating metrics
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels= []
            # predicted_onehot_labels_tk = [[] for _ in range(xconfig.top_num)]

            for batch_test in batches:
                x_batch_test_front,x_batch_test_behind, y_batch_test, y_batch_test_labels = zip(*batch_test)

                feed_dict = {
                    input_x_front: x_batch_test_front,
                    input_x_behind:x_batch_test_behind,
                    input_y: y_batch_test,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                batch_scores, cur_loss = sess.run([scores, loss], feed_dict)
                # print(batch_scores)

                # Prepare for calculating metrics
                for i in y_batch_test:
                    true_onehot_labels.append(i)
                for j in batch_scores:
                    predicted_onehot_scores.append(j)

                # Get the predicted labels by threshold
                batch_predicted_labels, batch_predicted_scores = \
                    dh.get_label(scores=batch_scores)

                # print('batch_predicted_labels:',batch_predicted_labels)

                # Add results to collection

                # print(batch_predicted_scores)
                for i in y_batch_test_labels:
                    # print('i:', i)
                    true_labels.append(i)
                for j in batch_predicted_labels:
                    # print('j:',j)
                    predicted_labels.append(j)
                for k in batch_predicted_scores:
                    predicted_scores.append(k)

                # print('predicted_labels:',predicted_labels)

                # Get onehot predictions by threshold
                batch_predicted_onehot_labels = \
                    dh.get_onehot_label(scores=batch_scores)
                for i in batch_predicted_onehot_labels:
                    predicted_onehot_labels.append(i)


                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1


                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1

            # Calculate Precision & Recall & F1 (threshold & topK)


            test_pre, test_rec, test_F, test_auc, test_prc=dh.evaluation_calculation(true_onehot_labels,predicted_onehot_labels,predicted_onehot_scores)
            test_loss = float(test_loss / test_counter)

            every_label_pre,every_label_rec,every_label_F=dh.evaluation_every_label(true_onehot_labels,predicted_onehot_labels,predicted_onehot_scores)

            logger.info("All Test Dataset: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(test_loss, test_auc, test_prc))

            # Predict by threshold
            logger.info("Predict : Precision {0:g}, Recall {1:g}, F1 {2:g}"
                        .format(test_pre, test_rec, test_F))

            # logger.info("Predict_every_label : ",str(every_label_pre), str(every_label_rec), str(every_label_F))
            label_id=0
            for pre,rec,f1 in zip(every_label_pre,every_label_rec,every_label_F):

                logger.info("Predict : label {0},Precision {1:g}, Recall {2:g}, F1 {3:g}"
                           .format(label_id,round(pre,4), round(rec,4), round(f1,4)))
                label_id+=1




            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", data_id=test_data.testid,
                                      all_labels=true_labels, all_predict_labels=predicted_labels,
                                      all_predict_scores=predicted_scores)

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    test_cnn()

