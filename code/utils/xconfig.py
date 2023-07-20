# -*- coding: utf-8 -*-
# @Time    : 2019-07-18 16:40
# @Author  : bo
# @File    : xconfig.py
# @Software: PyCharm

import os

# Data Parameters

prepre_dir=os.path.abspath(os.path.join(os.getcwd(), ".."))

# training_data_file = prepre_dir+"/test_data/train.json"

training_data_file = prepre_dir+"/Data/train_add_set.json"


validation_data_file =  prepre_dir+"/Data/test_add_set.json"
test_data_file=  prepre_dir+"/Data/test_add_set.json"

word2vec_file="/Users/liubo22/Downloads/fasttext_DW_wordvector.bin"
#
checkpoint_file =1580790756

# Model Hyperparameters
learning_rate=0.001
pad_seq_len=100

embedding_dim=100
embedding_type=1
fc_hidden_size=1024
filter_sizes=3,4,5
num_filters=128
dropout_keep_prob=0.5
l2_reg_lambda=0.0
num_classes=2
batch_size=128

num_epochs=20
evaluate_every=10

norm_ratio=2
# decay_steps=5000
decay_steps=50
decay_rate=0.95

# Misc Parameters
allow_soft_placement=True
log_device_placement=False
gpu_options_allow_growth=True



