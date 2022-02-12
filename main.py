#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = "suqiang@kuaishou.com"

import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,Model

BATCH_SIZE = 256
value_ranges = list(zip(range(0,12),[2000] * 10)) + [(12, 300)]

def make_parse_csv():
    def parse_csv(lines):
        fields_defaults = [int()] * 13 + [float()] * 13
        fields = tf.io.decode_csv(lines,fields_defaults)

        id_features = []
        for (index, _) in value_ranges:
            id_features.append(fields[index])
        
        float_features = [fields[i] for i in range(13, 25)]
        numeric = tf.stack(float_features,axis=1)
        features = tuple(id_features + numeric)
        labels = tf.clip_by_value(fields[25],clip_value_min = 0, clip_value_max = 1)
        return features,labels
    return parse_csv

def make_dataset(filelist, repeat_times=1):
    dataset = tf.data.TextLineDataset(filelist)
    dataset = dataset.map(make_parse_csv())
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(2 * BATCH_SIZE + 1)
    dataset = dataset.repeat(repeat_times)
    return dataset

train_files = ['./data/train_0.csv']
eval_files = ['./data/test.csv']

def D2Q():
    inputs=[]
    pxtr_sparse_concat = []
    for dim in list(zip(*value_ranges))[1]:
        if dim != 2000:
            continue
        x = layers.Input(shape=(1,),dtype=tf.int32)
        e = layers.Embedding(input_dim=dim, output_dim=32)(x)
        inputs.append(x)
        pxtr_sparse_layers.append(e)
    
    duration_concat = []
    x = layers.Input(shape=(1,),dtype=tf.int32)
    e = layers.Embedding(input_dim=dim, output_dim=32)(x)
    inputs.append(x)
    duration_concat.append(e)
    
    pxtr_dense_concat = []
    numeric = layers.Input(shape=(1,12),dtype=tf.float32)
    inputs.append(numeric)
    pxtr_dense_concat.append(numeric)
    
    wt_perc = layers.Input(shape=(1,),dtype=tf.float32)
    inputs.append(wt_perc)

    hidden_layer = []
    pxtr_sparse_emb = tf.concat(pxtr_sparse_concat,1)
    for dim in [512]:
        pxtr_sparse_emb = layers.Dense(dim,tf.nn.swish)(pxtr_sparse_emb)
    hidden_layer.append(pxtr_sparse_emb)
    
    duration_emb = tf.concat(duration_concat,1)
    for dim in [32]:
        pxtr_sparse_emb = layers.Dense(dim,tf.nn.swish)(duration_emb)
    hidden_layer.append(duration_emb)
    
    pxtr_dense_emb = tf.concat(pxtr_dense_concat,1)
    for dim in [32]:
        pxtr_dense_emb = layers.Dense(dim,tf.nn.swish)(pxtr_dense_emb)
    hidden_layer.append(pxtr_dense_emb)
    
    x = tf.concat(hidden_layer,1)
    
    for dim in [512,256,128,64]:
        x = layers.Dense(dim,tf.nn.swish)(x)
    out = layers.Dense(1, tf.sigmoid)(x)

    return Model(inputs = inputs, outputs = out)

def train():
    model = D2Q()
    loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),)
    model.fit(make_dataset(train_files,1))
    model.save_weights('./dumps/model/d2q_model')
    
if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()

