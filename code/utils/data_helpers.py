# -*- coding: utf-8 -*-
# @Time    : 2019-07-16 14:52
# @Author  : bo
# @File    : data_helpers.py
# @Software: PyCharm

#20190807修改

#20191108修改
#20191109修改

#数据处理、词向量加载等

import os
import heapq
import multiprocessing
import gensim
import logging
import json
import numpy as np
import jieba

from collections import OrderedDict
#from pylab import *

from tflearn.data_utils import pad_sequences,to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger



def get_onehot_label(scores):
    predicted_onehot_labels=[]
    scores=np.ndarray.tolist(scores)
    # print('scores:',scores)
    for score in scores:
        # print('score:',score)
        onehot_labels_list=[0]*len(score)
        max_score_index=score.index(max(score))
        onehot_labels_list[max_score_index]=1
        # print('onehot_labels_list:',onehot_labels_list)
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels







def get_label(scores):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`
    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    # print("scores:",scores)

    for score in scores:

        predicted_labels.append(score.index(max(score)))
        predicted_scores.append(max(score))
    return predicted_labels, predicted_scores

def load_data_and_labels(data_file,num_labels,vocab,data_aug_flag):
    # model=word2vec.Word2Vec.load(word2vec_file)

    data=data_word2vec(input_file=data_file,num_labels=num_labels,vocab=vocab)
    if data_aug_flag:
        data=data_augmented(data)
    return data




def data_word2vec(input_file,num_labels,vocab):
    # vocab=dict([(k,v.index) for (k,v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result=[]
        # print("vocab",vocab)
        for item in content:
            # if item not in ['发布','头条',"文章"]:
                # print("item:",item)
                try:  #20191012   {}--->TypeError: unhashable type: 'dict'
                    word2id=vocab.get(item)
                    print("item:",item)
                    if word2id is None:
                        word2id=0
                    result.append(word2id)
                except:
                    pass
        return result


    def _create_onehot_labels(labels_index):
        label=[0]*num_labels

        label[int(labels_index)] = 1


        return label

    with open(input_file,encoding='utf-8',errors='ignore') as fin:
        labels=[]
        front_testid=[]
        behind_testid=[]
        front_content_indexlist=[]
        behind_content_indexlist=[]
        total_line=0
        for eachline in fin:
            data=json.loads(eachline)
            front_testid.append(data['front_testid'])
            behind_testid.append(data['behind_testid'])
            labels.append(data["label"])

            front_content_indexlist.append(_token_to_index(data['front_features']))
            behind_content_indexlist.append(_token_to_index(data['behind_features']))
            total_line+=1






    class _Data:
        def __init__(self):
            pass
        @property
        def number(selfs):
            return total_line

        @property
        def front_testid(self):
            return front_testid
        @property
        def behind_testid(self):
            return behind_testid
        @property
        def front_tokenindex(self):
            return front_content_indexlist

        @property
        def behind_tokenindex(self):
            return behind_content_indexlist
        @property
        def labels(self):
            return labels

    return _Data()


def load_word2vec_matrix(embedding_size,word2vec_file,w2v_model,vocab):
    """
    Return the word2vec model matrix.
    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """

    vocab_size = len(vocab)
    # vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():


        if key is not None:

            embedding_matrix[value] = w2v_model[key]
    return vocab_size, embedding_matrix

def data_augmented(data, drop_rate=1.0):
    """
    Data augmented.
    Args:
        data: The Class Data()
        drop_rate: The drop rate
    Returns:
        aug_data
    """
    aug_num = data.number
    aug_testid = data.testid
    aug_tokenindex = data.tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels
    aug_labels_num = data.labels_num

    for i in range(len(data.tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_testid.append(data.testid[i])
            aug_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])
            aug_labels_num.append(data.labels_num[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_testid.append(data.testid[i])
                aug_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])
                aug_labels_num.append(data.labels_num[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def testid(self):
            return aug_testid

        @property
        def tokenindex(self):
            return aug_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def labels_num(self):
            return aug_labels_num

    return _AugData()



def pad_data(data,pad_seq_len):
    pad_seq_front=pad_sequences(data.front_tokenindex,maxlen=pad_seq_len,value=0.)
    pad_seq_behind=pad_sequences(data.behind_tokenindex,maxlen=pad_seq_len,value=0.)
    # onehot_labels=data.onehot_labels
    onthot_labels=to_categorical(data.labels,nb_classes=2)
    # return pad_seq,onehot_labels
    return pad_seq_front,pad_seq_behind,onthot_labels



def batch_iter(data,batch_size,num_epoches,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int((data_size-1)/batch_size)+1

    for epoch in range(num_epoches):
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_size*batch_num
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]



def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_scores):
    """
    Create the prediction file.
    Args:
        output_file: The all classes predicted results provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a <.json> file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)

        #print('data_id：',data_id)
        # print(all_predict_labels)
        for i in range(data_size):

            predict_labels = all_predict_labels[i]
            predict_scores =all_predict_scores[i]

            labels=all_labels[i]

            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_scores', predict_scores)
                #,
                #('text',text)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def evaluation_calculation(true_onehot_labels,predicted_onehot_labels,predicted_onehot_scores):

    test_pre = precision_score(y_true=np.array(true_onehot_labels),
                               y_pred=np.array(predicted_onehot_labels), average="macro")

    test_rec = recall_score(y_true=np.array(true_onehot_labels),
                            y_pred=np.array(predicted_onehot_labels), average='macro')
    test_F = f1_score(y_true=np.array(true_onehot_labels),
                      y_pred=np.array(predicted_onehot_labels), average='macro')

    # Calculate the average AUC
    test_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                             y_score=np.array(predicted_onehot_scores), average='macro')

    # Calculate the average PR
    test_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                       y_score=np.array(predicted_onehot_scores), average="macro")

    return test_pre,test_rec,test_F,test_auc,test_prc


def evaluation_every_label(true_onehot_labels, predicted_onehot_labels, predicted_onehot_scores):
    every_label_pre = precision_score(y_true=np.array(true_onehot_labels),
                               y_pred=np.array(predicted_onehot_labels), average=None)

    every_label_rec = recall_score(y_true=np.array(true_onehot_labels),
                            y_pred=np.array(predicted_onehot_labels), average=None)
    every_label_F = f1_score(y_true=np.array(true_onehot_labels),
                      y_pred=np.array(predicted_onehot_labels), average=None)





    return every_label_pre,every_label_rec,every_label_F

















































